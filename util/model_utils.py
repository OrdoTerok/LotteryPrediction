
# Standard library imports
import os
import json
import hashlib
import datetime
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import scipy.special
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression

# Local imports
import config
from data.loaders import fetch_data_from_datagov, load_data_from_kaggle
from data.preprocessing import combine_and_clean_data, save_to_file, prepare_data_for_lstm
from data.split import split_dataframe_by_percentage
from util.data_utils import analyze_value_ranges_per_ball
from util.plot_utils import (
    plot_multi_round_ball_distributions,
    plot_multi_round_powerball_distribution,
    plot_ball_distributions,
    plot_powerball_distribution
)
from util.plot_utils_std import (
    plot_multi_round_true_std,
    plot_multi_round_pred_std,
    plot_multi_round_kl_divergence
)
from util.experiment_tracker import ExperimentTracker
from util.metrics import smooth_labels, mix_uniform, kl_to_uniform, kl_divergence
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from models.mlp_model import MLPModel
from models.lgbm_model import LightGBMModel


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


def run_pipeline(config):
    """
    Orchestrates the full pipeline: data loading, meta-optimization, iterative stacking, evaluation, and plotting.
    """
    # Modular logging, experiment tracking, and cache are set up in main.py
    # Accept tracker and cache as arguments for modularity
    DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'
    from util.log_utils import get_logger, setup_logging
    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    logs_dir = os.path.abspath(logs_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # Create a unique log file for each run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(logs_dir, f'log_{timestamp}.rtf')
    setup_logging(log_filename)
    logger = get_logger()
    from util.cache import Cache
    cache = Cache()
    tracker = ExperimentTracker()
    logger.info("Starting the Powerball data download process...")
    datagov_df = fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = load_data_from_kaggle(config.KAGGLE_CSV_FILE)
    tracker.start_run({k: getattr(config, k) for k in dir(config) if k.isupper()})
    if not datagov_df.empty or not kaggle_df.empty:
        logger.info("Successfully loaded the Powerball data into a DataFrame.")
        logger.info(f"Data includes {kaggle_df.shape[0]} draws and has {kaggle_df.shape[1]} columns.")
    final_df = combine_and_clean_data(datagov_df, kaggle_df)
    save_to_file(final_df)
    tracker.log_artifact('data_sets/base_dataset.csv', artifact_name='base_dataset.csv')
    logger.info("--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
    analyze_value_ranges_per_ball(final_df)
    train_df, test_df = split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    logger.info(f"Data split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
    look_back_window = config.LOOK_BACK_WINDOW
    X_test, y_test = prepare_data_for_lstm(test_df, look_back=look_back_window)
    logger.info(f"Prepared testing data shape: {X_test.shape}")
    if X_test.size == 0:
        logger.error("Not enough data to create test sequences. Exiting.")
        tracker.end_run()
        return float('inf')
    y_true_first_five = np.argmax(y_test[0], axis=-1) + 1
    y_true_sixth = np.argmax(y_test[1], axis=-1) + 1
    # ...existing code...
    run_meta_optimization(final_df, config)
    prev_pred_first_five = None
    prev_pred_sixth = None
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            if isinstance(history, list) and len(history) > 0:
                prev_results = history[-1]
                prev_pred_first_five = np.array(prev_results.get('first_five_pred_numbers'))
                prev_pred_sixth = np.array(prev_results.get('sixth_pred_number'))
        except Exception:
            pass
    rounds_first_five, rounds_sixth, round_labels = run_iterative_stacking(
        train_df, test_df, config, y_true_first_five, y_true_sixth,
        prev_pred_first_five=prev_pred_first_five, prev_pred_sixth=prev_pred_sixth
    )
    # Log predictions artifact
    if os.path.exists(history_path):
        tracker.log_artifact(history_path)
    if rounds_first_five:
        plot_multi_round_ball_distributions(
            y_true=y_true_first_five,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            n_classes=69,
            title_prefix='Ball',
            round_labels=round_labels,
            prev_label='Previous'
        )
        tracker.log_artifact('multi_round_ball_distributions.png') if os.path.exists('multi_round_ball_distributions.png') else None
    if rounds_sixth:
        plot_multi_round_powerball_distribution(
            y_true=y_true_sixth,
            rounds_pred_list=rounds_sixth,
            prev_pred=prev_pred_sixth,
            n_classes=26,
            title='Powerball (6th Ball) Distribution',
            round_labels=round_labels,
            prev_label='Previous'
        )
        tracker.log_artifact('multi_round_powerball_distribution.png') if os.path.exists('multi_round_powerball_distribution.png') else None
        plot_multi_round_true_std(
            y_true=y_true_first_five,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            round_labels=round_labels,
            prev_label='Previous'
        )
        tracker.log_artifact('multi_round_true_std.png') if os.path.exists('multi_round_true_std.png') else None
        plot_multi_round_pred_std(
            y_true=y_true_first_five,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            round_labels=round_labels,
            prev_label='Previous'
        )
        tracker.log_artifact('multi_round_pred_std.png') if os.path.exists('multi_round_pred_std.png') else None
        plot_multi_round_kl_divergence(
            y_true=y_true_first_five,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            n_classes=69,
            round_labels=round_labels,
            prev_label='Previous'
        )
        tracker.log_artifact('multi_round_kl_divergence.png') if os.path.exists('multi_round_kl_divergence.png') else None
    # Example: log a metric (extend as needed)
    # tracker.log_metric('example_metric', 0.0)
    tracker.end_run()

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
        (0.0, 1.0),  # JACCARD_LOSS_WEIGHT
        (0.0, 1.0),  # DUPLICATE_PENALTY_WEIGHT
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
        # Run the pipeline for this round
        # Modular: pipeline handles model training and evaluation
        run_pipeline(config)
        # Diagnostics: append predictions if available
        try:
            history = get_results_history()
            if isinstance(history, list) and len(history) > 0:
                results = history[-1]
                ff_pred = np.asarray(results['first_five_pred_numbers'])
                s_pred = np.asarray(results['sixth_pred_number'])
                rounds_first_five.append(ff_pred)
                rounds_sixth.append(s_pred)
        except Exception as e:
            logger.error(f"[Iterative Stacking] Could not append predictions for round {round_idx+1}: {e}")
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
            print(f"  Model {idx} ({type(m)}): Exception during prediction: {e}")
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
