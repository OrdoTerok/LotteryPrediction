def cross_validate_model(model, X, y, cv=5, **kwargs):
    """
    Utility to perform cross-validation using a model's cross_validate method.
    Returns list of per-fold evaluation results.
    """
    if hasattr(model, 'cross_validate'):
        return model.cross_validate(X, y, cv=cv, **kwargs)
    else:
        raise NotImplementedError(f"Model {type(model)} does not support cross-validation.")

# Standard library imports
import os
import json
import hashlib

# Third-party imports
import numpy as np
import tensorflow as tf
try:
    from tensorflow import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
except ImportError:
    pass

# Local imports
import config
from data.loaders import fetch_data_from_datagov, load_data_from_kaggle
from data.preprocessing import combine_and_clean_data, save_to_file, prepare_data_for_lstm
from data.split import split_dataframe_by_percentage
from util.plot_utils import (
    plot_multi_round_ball_distributions,
    plot_multi_round_powerball_distribution
)
from util.plot_utils_std import (
    plot_multi_round_true_std,
    plot_multi_round_pred_std,
    plot_multi_round_kl_divergence
)
from util.experiment_tracker import ExperimentTracker


def get_results_history(cache=None):
    """
    Loads and caches the results_predictions_history.json file in memory or via cache utility.
    Returns the cached history on subsequent calls.
    """
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    if cache is not None:
        history = cache.get(history_path)
        if history is not None:
            return history
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    else:
        history = []
    if cache is not None:
        cache.set(history_path, history)
    return history


def run_pipeline(config, from_iterative_stacking=False, cv=None):
    """
    Orchestrates the full pipeline: data loading, meta-optimization, iterative stacking, evaluation, and plotting.
    """
    # Modular logging, experiment tracking, and cache are set up in main.py
    # Accept tracker and cache as arguments for modularity
    DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'
    from util.log_utils import get_logger
    logger = get_logger()
    from util.cache import Cache
    cache = Cache()
    tracker = ExperimentTracker()
    # High-level log: pipeline data preparation and validation
    # Cache key based on input file modification times
    kaggle_path = config.KAGGLE_CSV_FILE
    datagov_path = 'data_sets/datagov_cache.csv'
    cache_key = f"combined_df_{os.path.getmtime(kaggle_path)}"
    cached_df = cache.get(cache_key)
    if not from_iterative_stacking:
        tracker.start_run({k: getattr(config, k) for k in dir(config) if k.isupper()})
    if cached_df is not None:
        final_df = cached_df
    else:
        datagov_df = fetch_data_from_datagov(DATAGOV_API_URL)
        kaggle_df = load_data_from_kaggle(kaggle_path)
        final_df = combine_and_clean_data(datagov_df, kaggle_df)
        cache.set(cache_key, final_df)
        save_to_file(final_df)
        if not from_iterative_stacking:
            tracker.log_artifact('data_sets/base_dataset.csv', artifact_name='base_dataset.csv')
    train_df, test_df = split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    look_back_window = config.LOOK_BACK_WINDOW
    X_test, y_test = prepare_data_for_lstm(test_df, look_back=look_back_window)
    if X_test.size == 0:
        logger.error("Not enough data to create test sequences. Exiting.")
        tracker.end_run()
        return float('inf')
    y_true_first_five = np.argmax(y_test[0], axis=-1) + 1
    y_true_sixth = np.argmax(y_test[1], axis=-1) + 1
    logger.info("[Pipeline] Running Meta Optimization")
    run_meta_optimization(final_df, config)
    logger.info("[Pipeline] Meta Optimization complete.")
    prev_pred_first_five = None
    prev_pred_sixth = None
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            if isinstance(history, list) and len(history) > 0:
                prev_results = history[-1]
                prev_pred_first_five = np.array(prev_results.get('first_five_pred_numbers'), dtype=np.float32)
                prev_pred_sixth = np.array(prev_results.get('sixth_pred_number'), dtype=np.float32)
        except Exception:
            pass
    from models.model_factory import get_model
    models = []
    X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
    # LSTM
    # Ensure input_shape is always (timesteps, features) for LSTM
    if X_train[0].ndim == 2:
        input_shape = X_train[0].shape
    elif X_train[0].ndim == 3:
        input_shape = X_train[0].shape[1:]
    else:
        raise ValueError(f"Unexpected X_train[0] shape: {X_train[0].shape}")
    lstm_model = get_model(
        'lstm',
        input_shape=input_shape,
        hp=None,
        use_custom_loss=True,
        force_low_units=False,
        force_simple=False,
        units=getattr(config, 'LSTM_UNITS', 64),
        dropout=getattr(config, 'LSTM_DROPOUT', 0.2),
        learning_rate=getattr(config, 'LSTM_LEARNING_RATE', 1e-3),
        label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.0),
        temp_max=getattr(config, 'TEMP_MAX', 0.0)
    )
    if cv is None:
        cv = getattr(config, 'CV_FOLDS', 1)
    if cv > 1:
        logger.info(f"[Pipeline] Running cross-validation for LSTM (cv={cv})")
        lstm_cv_results = cross_validate_model(lstm_model, X_train, {'first_five': y_train[0], 'sixth': y_train[1]}, cv=cv, epochs=3, batch_size=32, verbose=0)
        logger.info(f"[Pipeline] LSTM CV results: {lstm_cv_results}")
    else:
        lstm_model.fit(
            X_train,
            {'first_five': y_train[0], 'sixth': y_train[1]},
            epochs=3, batch_size=32, verbose=0
        )
    models.append(lstm_model)
    # RNN
    # Ensure input_shape is always (timesteps, features) for RNN
    if X_train[0].ndim == 2:
        rnn_input_shape = X_train[0].shape
    elif X_train[0].ndim == 3:
        rnn_input_shape = X_train[0].shape[1:]
    else:
        raise ValueError(f"Unexpected X_train[0] shape: {X_train[0].shape}")
    rnn_model = get_model(
        'rnn',
        input_shape=rnn_input_shape,
        hp=None,
        units=getattr(config, 'RNN_UNITS', 64),
        dropout=getattr(config, 'RNN_DROPOUT', 0.2),
        learning_rate=getattr(config, 'RNN_LEARNING_RATE', 1e-3),
        label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.0),
        temp_max=getattr(config, 'TEMP_MAX', 0.0)
    )
    if cv > 1:
        logger.info(f"[Pipeline] Running cross-validation for RNN (cv={cv})")
        rnn_cv_results = cross_validate_model(rnn_model, X_train, {'first_five': y_train[0], 'sixth': y_train[1]}, cv=cv, epochs=3, batch_size=32, verbose=0)
        logger.info(f"[Pipeline] RNN CV results: {rnn_cv_results}")
    else:
        rnn_model.fit(
            X_train,
            {'first_five': y_train[0], 'sixth': y_train[1]},
            epochs=3, batch_size=32, verbose=0
        )
    models.append(rnn_model)
    # MLP
    mlp_input_shape = (X_train.shape[1] * X_train.shape[2],)
    mlp_model = get_model(
        'mlp',
        input_shape=mlp_input_shape,
        hidden_units=getattr(config, 'MLP_HIDDEN_UNITS', 64),
        dropout_rate=getattr(config, 'MLP_DROPOUT', 0.5),
        learning_rate=getattr(config, 'MLP_LEARNING_RATE', 1e-3),
        label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.0),
        temp_max=getattr(config, 'TEMP_MAX', 0.0)
    )
    # Flatten X_train for MLP: (batch, timesteps, features) -> (batch, timesteps * features)
    X_train_mlp = X_train.reshape(X_train.shape[0], -1)
    if cv > 1:
        logger.info(f"[Pipeline] Running cross-validation for MLP (cv={cv})")
        mlp_cv_results = cross_validate_model(mlp_model, X_train_mlp, {'first_five': y_train[0], 'sixth': y_train[1]}, cv=cv, epochs=3, batch_size=32, verbose=0)
        logger.info(f"[Pipeline] MLP CV results: {mlp_cv_results}")
    else:
        mlp_model.fit(
            X_train_mlp,
            {'first_five': y_train[0], 'sixth': y_train[1]},
            epochs=3, batch_size=32, verbose=0
        )
    models.append(mlp_model)
    # LightGBM
    lgbm_params = {
    'num_leaves': int(getattr(config, 'LGBM_NUM_LEAVES', 31)),
    'learning_rate': float(getattr(config, 'LGBM_LEARNING_RATE', 0.1)),
    'max_depth': int(getattr(config, 'LGBM_MAX_DEPTH', -1))
    }
    lgbm_model = get_model(
        'lgbm',
        num_first=5,
        num_first_classes=69,
        num_sixth_classes=26,
        params=lgbm_params
    )
    # Flatten X_train for LightGBM: (batch, timesteps, features) -> (batch, timesteps * features)
    X_train_lgbm = X_train.reshape(X_train.shape[0], -1)
    if cv > 1:
        logger.info(f"[Pipeline] Running cross-validation for LightGBM (cv={cv})")
        lgbm_cv_results = cross_validate_model(lgbm_model, X_train_lgbm, (y_train[0], y_train[1]), cv=cv)
        logger.info(f"[Pipeline] LightGBM CV results: {lgbm_cv_results}")
    else:
        lgbm_model.fit(X_train_lgbm, (y_train[0], y_train[1]))
    models.append(lgbm_model)
    # Ensemble predictions
    # Stack X_test into a 3D array if it is a list of 2D arrays
    if isinstance(X_test, list) or (hasattr(X_test, 'ndim') and X_test.ndim == 1 and hasattr(X_test[0], 'shape')):
        X_test_batch = np.stack(X_test, axis=0)
    else:
        X_test_batch = X_test
    ensemble_first, ensemble_sixth = ensemble_predict(models, X_test_batch)
    logger.info(f"[Ensemble] First five shape: {ensemble_first.shape}, Sixth shape: {ensemble_sixth.shape}")
    if not from_iterative_stacking:
        logger.info("[Pipeline] Running Iterative Stacking")
        rounds_first_five, rounds_sixth, round_labels = run_iterative_stacking(
            train_df, test_df, config, y_true_first_five, y_true_sixth,
            prev_pred_first_five=prev_pred_first_five, prev_pred_sixth=prev_pred_sixth
        )
        logger.info("[Pipeline] Iterative Stacking complete.")
    else:
        # If called from iterative stacking, do not recurse
        rounds_first_five, rounds_sixth, round_labels = [], [], []
    if not from_iterative_stacking:
        # Log predictions artifact and plots before returning
        if os.path.exists(history_path):
            tracker.log_artifact(history_path)
        def log_plot_and_artifact(plot_func, plot_args, artifact_path):
            plot_func(**plot_args)
            if os.path.exists(artifact_path):
                tracker.log_artifact(artifact_path)

        if rounds_first_five:
            log_plot_and_artifact(
                plot_multi_round_ball_distributions,
                dict(
                    y_true=y_true_first_five,
                    rounds_pred_list=rounds_first_five,
                    prev_pred=prev_pred_first_five,
                    num_balls=5,
                    n_classes=69,
                    title_prefix='Ball',
                    round_labels=round_labels,
                    prev_label='Previous'
                ),
                'multi_round_ball_distributions.png'
            )
        if rounds_sixth:
            log_plot_and_artifact(
                plot_multi_round_powerball_distribution,
                dict(
                    y_true=y_true_sixth,
                    rounds_pred_list=rounds_sixth,
                    prev_pred=prev_pred_sixth,
                    n_classes=26,
                    title='Powerball (6th Ball) Distribution',
                    round_labels=round_labels,
                    prev_label='Previous'
                ),
                'multi_round_powerball_distribution.png'
            )
            log_plot_and_artifact(
                plot_multi_round_true_std,
                dict(
                    y_true=y_true_first_five,
                    rounds_pred_list=rounds_first_five,
                    prev_pred=prev_pred_first_five,
                    num_balls=5,
                    round_labels=round_labels,
                    prev_label='Previous'
                ),
                'multi_round_true_std.png'
            )
            log_plot_and_artifact(
                plot_multi_round_pred_std,
                dict(
                    y_true=y_true_first_five,
                    rounds_pred_list=rounds_first_five,
                    prev_pred=prev_pred_first_five,
                    num_balls=5,
                    round_labels=round_labels,
                    prev_label='Previous'
                ),
                'multi_round_pred_std.png'
            )
            log_plot_and_artifact(
                plot_multi_round_kl_divergence,
                dict(
                    y_true=y_true_first_five,
                    rounds_pred_list=rounds_first_five,
                    prev_pred=prev_pred_first_five,
                    num_balls=5,
                    n_classes=69,
                    round_labels=round_labels,
                    prev_label='Previous'
                ),
                'multi_round_kl_divergence.png'
            )
        # Example: log a metric (extend as needed)
        # tracker.log_metric('example_metric', 0.0)
        tracker.end_run()
        return ensemble_first, ensemble_sixth
    else:
        return ensemble_first, ensemble_sixth

def run_meta_optimization(final_df, config):
    """
    Run meta-parameter optimization (PSO or Bayesian) and update config with best values.
    """
    var_names = [
        "LABEL_SMOOTHING",
        "TEMP_MAX",
        "EARLY_STOPPING_PATIENCE",
        "OVERCOUNT_PENALTY_WEIGHT",
        "ENTROPY_PENALTY_WEIGHT",
        "JACCARD_LOSS_WEIGHT",
        "DUPLICATE_PENALTY_WEIGHT",
        "ANTI_COPY_PENALTY_WEIGHT",
        "LGBM_NUM_LEAVES",
        "LGBM_LEARNING_RATE",
        "LGBM_MAX_DEPTH"
    ]
    bounds = [
    (0.0, 0.3),
    (0.0, 0.3),
    (0.5, 1.5),
    (1.5, 2.5),
    (1, 10),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 2.0),  # ANTI_COPY_PENALTY_WEIGHT (tune from 0 to 2)
    (7, 127),
    (0.01, 0.3),
    (3, 12)
    ]
    from optimization.meta_search import MetaParameterSearch
    from util.log_utils import get_logger
    logger = get_logger()
    from data.data_handler import DataHandler
    train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    meta_search = MetaParameterSearch(method=getattr(config, 'META_OPT_METHOD', 'pso'))
    best = meta_search.search(
        var_names,
        bounds,
        (train_df, test_df),
        n_trials=getattr(config, 'PSO_ITER', 10),
        n_particles=getattr(config, 'PSO_PARTICLES', 5),
        n_iter=getattr(config, 'PSO_ITER', 10)
    )
    if best is None:
        logger.error("Meta-optimization failed or was aborted (e.g., due to recursion guard). Skipping meta-parameter update.")
        return
    logger.info(f"Best meta-hyperparameters ({getattr(config, 'META_OPT_METHOD', 'pso')}): %s", dict(zip(var_names, best)))
    for i, name in enumerate(var_names):
        setattr(config, name, best[i])

import os
def run_iterative_stacking(train_df, test_df, config, y_true_first_five, y_true_sixth, prev_pred_first_five=None, prev_pred_sixth=None):
    from util.log_utils import get_logger
    logger = get_logger()
    rounds_first_five = []
    rounds_sixth = []
    round_labels = []
    num_rounds = getattr(config, 'ITERATIVE_STACKING_ROUNDS', 1) if getattr(config, 'ITERATIVE_STACKING', False) else 1
    meta_cols = [f'prev_pred_ball_{j+1}' for j in range(5)] + ['prev_pred_sixth', 'is_pseudo']
    base_train_df = train_df.copy()
    for col in meta_cols:
        if col not in base_train_df.columns:
            base_train_df = base_train_df.assign(**{col: np.zeros(len(base_train_df), dtype=np.float32)})
    for col in meta_cols:
        if col not in test_df.columns:
            test_df = test_df.assign(**{col: np.zeros(len(test_df), dtype=np.float32)})
    test_df = test_df[base_train_df.columns]
    best_hp_lstm = None
    best_hp_rnn = None
    best_hp_mlp = None
    for round_idx in range(num_rounds):
        noise_std = 0.5
        meta_feature_cols = [f'prev_pred_ball_{j+1}' for j in range(5)] + ['prev_pred_sixth']
        if 'is_pseudo' in base_train_df.columns:
            mask = base_train_df['is_pseudo'] == 0
            for col in meta_feature_cols:
                try:
                    base_train_df.loc[mask, col] += np.random.normal(0, noise_std, mask.sum())
                except Exception as fw:
                    logger.warning(f"Noise addition warning: {fw}")

        # Run the pipeline for this round and capture ensemble predictions
        logger.info(f"[Iterative Stacking] Running pipeline for round {round_idx+1}...")
        ensemble_first, ensemble_sixth = run_pipeline(config, from_iterative_stacking=True)
        logger.info(f"[Iterative Stacking] Pipeline complete for round {round_idx+1}. Ensemble predictions captured.")

        # Diagnostics: append ensemble predictions directly
        if ensemble_first is not None and ensemble_sixth is not None:
            rounds_first_five.append(np.asarray(ensemble_first))
            rounds_sixth.append(np.asarray(ensemble_sixth))
        else:
            logger.warning(f"[Iterative Stacking] Ensemble predictions missing for round {round_idx+1}.")

        # --- LOGICAL INTEGRATION OF ENSEMBLE PREDICTIONS AS META-FEATURES ---
        # Use the ensemble predictions to update meta-features in base_train_df for the next round
        # (Assume base_train_df has the same order as the training data used in run_pipeline)
        if round_idx < num_rounds - 1 and ensemble_first is not None and ensemble_sixth is not None:
            # For each training sample, update prev_pred_ball_1..5 and prev_pred_sixth
            # Use argmax to get predicted class (ball number) for each sample and ball
            try:
                pred_balls = np.argmax(ensemble_first, axis=-1) + 1  # shape: (n_samples, 5)
                pred_sixth = np.argmax(ensemble_sixth, axis=-1) + 1  # shape: (n_samples, 1) or (n_samples,)
                for j in range(5):
                    base_train_df[f'prev_pred_ball_{j+1}'] = pred_balls[:, j]
                # Handle shape for sixth ball
                if len(pred_sixth.shape) > 1 and pred_sixth.shape[1] == 1:
                    base_train_df['prev_pred_sixth'] = pred_sixth[:, 0]
                else:
                    base_train_df['prev_pred_sixth'] = pred_sixth
                logger.info(f"[Iterative Stacking] Updated meta-features in base_train_df for next round using ensemble predictions.")
            except Exception as e:
                logger.warning(f"[Iterative Stacking] Failed to update meta-features with ensemble predictions: {e}")

        logger.info(f"[Iterative Stacking] Completed round {round_idx+1}/{num_rounds}.")
        round_labels.append(f'Round {round_idx+1}')
    return rounds_first_five, rounds_sixth, round_labels

def ensemble_predict(models, X):
    """
    Ensemble predictions from multiple models using the strategy specified in config.ENSEMBLE_STRATEGY.
    Supported strategies:
      - 'average': simple mean of model predictions
      - 'weighted': weighted average (equal weights by default)
      - 'stacking': meta-learner (logistic regression) stacking
    """
    preds_first = []
    preds_sixth = []
    shapes_first = []
    shapes_sixth = []
    for idx, m in enumerate(models):
        try:
            pf, ps = m.predict(X, verbose=0)
            preds_first.append(pf)
            preds_sixth.append(ps)
            shapes_first.append(pf.shape)
            shapes_sixth.append(ps.shape)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"  Model {idx} ({type(m)}): Exception during prediction: {e}")
            preds_first.append(None)
            preds_sixth.append(None)
            shapes_first.append(None)
            shapes_sixth.append(None)
    valid_first = [s for s in shapes_first if s is not None]
    valid_sixth = [s for s in shapes_sixth if s is not None]
    import logging
    logger = logging.getLogger(__name__)
    if len(set(valid_first)) > 1 or len(set(valid_sixth)) > 1:
        logger.error(f"[ERROR][ensemble_predict] Prediction shape mismatch! first_five shapes: {shapes_first}, sixth shapes: {shapes_sixth}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch! See diagnostic output above.")

    strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'average').lower()
    if any(pf is None for pf in preds_first) or any(ps is None for ps in preds_sixth):
        logger.error("[ERROR][ensemble_predict] At least one prediction is None! Indices: %s %s", [i for i,pf in enumerate(preds_first) if pf is None], [i for i,ps in enumerate(preds_sixth) if ps is None])
        raise RuntimeError("[ensemble_predict] At least one prediction is None! See diagnostic output above.")
    valid_first_shapes = [pf.shape for pf in preds_first]
    valid_sixth_shapes = [ps.shape for ps in preds_sixth]
    if len(set(valid_first_shapes)) > 1 or len(set(valid_sixth_shapes)) > 1:
        logger.error(f"[ERROR][ensemble_predict] Prediction shape mismatch before aggregation! first_five shapes: {valid_first_shapes}, sixth shapes: {valid_sixth_shapes}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch before aggregation! See diagnostic output above.")
    if strategy == 'average':
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth
    elif strategy == 'weighted':
        weights = np.ones(len(models)) / len(models)
        weighted_first = np.tensordot(weights, np.array(preds_first), axes=1)
        weighted_sixth = np.tensordot(weights, np.array(preds_sixth), axes=1)
        return weighted_first, weighted_sixth
    elif strategy == 'stacking':
        from models.nn_meta_learner import NNMetaLearner
        n_samples, n_balls, n_classes = preds_first[0].shape
        stacked_first = np.zeros((n_samples, n_balls, n_classes))
        # Hyperparameter grid for meta-learner
        hidden_units_grid = [16, 32]
        epochs_grid = [5, 10]
        for b in range(n_balls):
            X_stack = np.stack([pf[:, b, :] for pf in preds_first], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_first, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            best_acc = -1
            best_pred = None
            for hu in hidden_units_grid:
                for ep in epochs_grid:
                    meta = NNMetaLearner(input_dim=len(models), output_dim=n_classes, hidden_units=hu, epochs=ep)
                    try:
                        meta.fit(X_stack, y_stack)
                        preds = meta.predict_proba(X_stack)
                        acc = np.mean(np.argmax(preds, axis=1) == y_stack)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = preds
                    except Exception:
                        continue
            if best_pred is not None:
                stacked_first[:, b, :] = best_pred.reshape(n_samples, n_classes)
            else:
                stacked_first[:, b, :] = np.mean([pf[:, b, :] for pf in preds_first], axis=0)
        n_samples, n_balls, n_classes = preds_sixth[0].shape
        stacked_sixth = np.zeros((n_samples, n_balls, n_classes))
        for b in range(n_balls):
            X_stack = np.stack([ps[:, b, :] for ps in preds_sixth], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_sixth, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            best_acc = -1
            best_pred = None
            for hu in hidden_units_grid:
                for ep in epochs_grid:
                    meta = NNMetaLearner(input_dim=len(models), output_dim=n_classes, hidden_units=hu, epochs=ep)
                    try:
                        meta.fit(X_stack, y_stack)
                        preds = meta.predict_proba(X_stack)
                        acc = np.mean(np.argmax(preds, axis=1) == y_stack)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = preds
                    except Exception:
                        continue
            if best_pred is not None:
                stacked_sixth[:, b, :] = best_pred.reshape(n_samples, n_classes)
            else:
                stacked_sixth[:, b, :] = np.mean([ps[:, b, :] for ps in preds_sixth], axis=0)
        return stacked_first, stacked_sixth
    else:
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth

def train_model(model, X, y, epochs=3, batch_size=32, validation_split=0.1, verbose=0, callbacks=None):
    cb = callbacks if callbacks is not None else []
    return model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=cb)

import hashlib
import numpy as np

def model_weights_hash(model):
    weights = model.get_weights()
    flat = np.concatenate([w.flatten() for w in weights if w.size > 0])
    return hashlib.md5(flat.tobytes()).hexdigest() if flat.size > 0 else 'empty'

def diversity_penalty(y_pred):
    # y_pred: (batch, classes)
    # Penalize repeated predictions in the batch
    y_pred_labels = np.argmax(y_pred, axis=-1)
    unique, counts = np.unique(y_pred_labels, return_counts=True)
    penalty = np.sum(counts[counts > 1] - 1) / len(y_pred_labels)
    return penalty
