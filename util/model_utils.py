def run_pipeline(config):
    """
    Orchestrates the full pipeline: data loading, meta-optimization, iterative stacking, evaluation, and plotting.
    """
    from data.data_handler import DataHandler
    import numpy as np
    import os, json
    from util.data_utils import analyze_value_ranges_per_ball
    from util.plot_utils import plot_multi_round_ball_distributions, plot_multi_round_powerball_distribution
    from util.plot_utils_std import (
        plot_multi_round_true_std,
        plot_multi_round_pred_std,
        plot_multi_round_kl_divergence
    )
    from util.experiment_tracker import ExperimentTracker
    DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'
    tracker = ExperimentTracker()
    print("Starting the Powerball data download process...")
    datagov_df = DataHandler.fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = DataHandler.load_data_from_kaggle(config.KAGGLE_CSV_FILE)
    # Track config at the start of the run
    tracker.start_run({k: getattr(config, k) for k in dir(config) if k.isupper()})
    if not datagov_df.empty or not kaggle_df.empty:
        print("\nSuccessfully loaded the Powerball data into a DataFrame.")
        print(f"Data includes {kaggle_df.shape[0]} draws and has {kaggle_df.shape[1]} columns.")
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        DataHandler.save_to_file(final_df)
        tracker.log_artifact('data_sets/base_dataset.csv', artifact_name='base_dataset.csv')
        print("\n--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
        analyze_value_ranges_per_ball(final_df)
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
        look_back_window = config.LOOK_BACK_WINDOW
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        print(f"Prepared testing data shape: {X_test.shape}")
        if X_test.size == 0:
            print("\nNot enough data to create test sequences. Exiting.")
            tracker.end_run()
            return
        y_true_first_five = np.argmax(y_test[0], axis=-1) + 1
        y_true_sixth = np.argmax(y_test[1], axis=-1) + 1
        from util.model_utils import run_meta_optimization, run_iterative_stacking
        run_meta_optimization(final_df, config)
        prev_pred_first_five = None
        prev_pred_sixth = None
        if os.path.exists('results_predictions.json'):
            try:
                with open('results_predictions.json', 'r') as f:
                    prev_results = json.load(f)
                prev_pred_first_five = np.array(prev_results.get('first_five_pred_numbers'))
                prev_pred_sixth = np.array(prev_results.get('sixth_pred_number'))
            except Exception:
                pass
        rounds_first_five, rounds_sixth, round_labels = run_iterative_stacking(
            train_df, test_df, config, y_true_first_five, y_true_sixth,
            prev_pred_first_five=prev_pred_first_five, prev_pred_sixth=prev_pred_sixth
        )
        # Log predictions artifact
        if os.path.exists('results_predictions.json'):
            tracker.log_artifact('results_predictions.json')
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
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")

def run_meta_optimization(final_df, config):
    """
    Run meta-parameter optimization (PSO or Bayesian) and update config with best values.
    """
    var_names = [
        "LABEL_SMOOTHING",
        "UNIFORM_MIX_PROB",
        "TEMP_MIN",
        "TEMP_MAX",
        "EARLY_STOPPING_PATIENCE",
        "OVERCOUNT_PENALTY_WEIGHT",
        "ENTROPY_PENALTY_WEIGHT",
        "JACCARD_LOSS_WEIGHT",
        "DUPLICATE_PENALTY_WEIGHT",
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
        (7, 127),
        (0.01, 0.3),
        (3, 12)
    ]
    if getattr(config, 'META_OPT_METHOD', 'pso').lower() == 'bayesian':
        from meta_tuning.bayesian_opt import bayesian_optimize
        best = bayesian_optimize(var_names, bounds, final_df, n_trials=getattr(config, 'PSO_ITER', 10))
        print("Best meta-hyperparameters (Bayesian):", dict(zip(var_names, best)))
    else:
        from meta_tuning.particle_swarm import particle_swarm_optimize
        best = particle_swarm_optimize(var_names, bounds, final_df, n_particles=config.PSO_PARTICLES, n_iter=config.PSO_ITER)
        print("Best meta-hyperparameters (PSO):", dict(zip(var_names, best)))
    for i, name in enumerate(var_names):
        setattr(config, name, best[i])

def run_iterative_stacking(train_df, test_df, config, y_true_first_five, y_true_sixth, prev_pred_first_five=None, prev_pred_sixth=None):
    """
    Run multi-round iterative stacking and pseudo-labeling. Returns (rounds_first_five, rounds_sixth, round_labels).
    """
    import json
    from util.plot_utils import plot_multi_round_ball_distributions
    import numpy as np
    import pandas as pd
    import os
    rounds_first_five = []
    rounds_sixth = []
    round_labels = []
    num_rounds = getattr(config, 'ITERATIVE_STACKING_ROUNDS', 1) if getattr(config, 'ITERATIVE_STACKING', False) else 1
    pseudo_train_df = None
    PSEUDO_CONFIDENCE_THRESHOLD = 0.9
    PSEUDO_MAX_SAMPLES = 100
    base_train_df = train_df.copy()
    best_hp_lstm = None
    best_hp_rnn = None
    best_hp_mlp = None
    for round_idx in range(num_rounds):
        print(f"\n[ITERATIVE STACKING] === Round {round_idx+1} of {num_rounds} ===")
        try:
            if round_idx > 0 and pseudo_train_df is not None:
                with open('results_predictions.json', 'r') as f:
                    results = json.load(f)
                pseudo_test_df = pseudo_train_df['test_df']
                first_five_pred = np.array(results['first_five_pred_numbers'])
                sixth_pred = np.array(results['sixth_pred_number'])
                first_five_softmax = np.array(results.get('first_five_pred_softmax')) if 'first_five_pred_softmax' in results else None
                sixth_softmax = np.array(results.get('sixth_pred_softmax')) if 'sixth_pred_softmax' in results else None
                n = min(len(pseudo_test_df), len(first_five_pred), len(sixth_pred))
                if first_five_softmax is not None:
                    n = min(n, len(first_five_softmax))
                if sixth_softmax is not None:
                    n = min(n, len(sixth_softmax))
                pseudo_labels = []
                pseudo_indices = []
                entropies_first = []
                entropies_sixth = []
                debug_conf_thresh = 0.5
                debug_entropy_thresh = 0.0
                for i in range(n):
                    accept = True
                    if first_five_softmax is not None and sixth_softmax is not None:
                        conf_first = np.max(first_five_softmax[i], axis=1)
                        conf_sixth = np.max(sixth_softmax[i], axis=1)
                        entropy_first = -np.sum(first_five_softmax[i] * np.log(first_five_softmax[i] + 1e-8), axis=1)
                        entropy_sixth = -np.sum(sixth_softmax[i] * np.log(sixth_softmax[i] + 1e-8), axis=1)
                        entropies_first.append(entropy_first)
                        entropies_sixth.append(entropy_sixth)
                        if not (np.all(conf_first > debug_conf_thresh) and np.all(conf_sixth > debug_conf_thresh)):
                            accept = False
                        if np.any(entropy_first < debug_entropy_thresh) or np.any(entropy_sixth < debug_entropy_thresh):
                            accept = False
                    if accept:
                        pseudo_indices.append(i)
                if entropies_first:
                    entropies_first = np.stack(entropies_first)
                    entropies_sixth = np.stack(entropies_sixth)
                np.random.shuffle(pseudo_indices)
                pseudo_indices = pseudo_indices[:min(PSEUDO_MAX_SAMPLES, n, len(pseudo_indices))]
                for i in pseudo_indices:
                    balls = first_five_pred[i].tolist() + sixth_pred[i].tolist()
                    pseudo_labels.append(' '.join(str(int(b)) for b in balls))
                pseudo_df = pseudo_test_df.iloc[pseudo_indices].copy()
                pseudo_df['Winning Numbers'] = pseudo_labels
                all_pseudo_numbers = np.concatenate([first_five_pred[pseudo_indices], sixth_pred[pseudo_indices]], axis=1).flatten()
                unique, counts = np.unique(all_pseudo_numbers, return_counts=True)
                print(f"[Pseudo-Labeling] Distribution of pseudo-labeled numbers: {dict(zip(unique, counts))}")
                base_train_df = pd.concat([base_train_df, pseudo_df], ignore_index=True)
                base_train_df = base_train_df.sample(frac=1, random_state=round_idx).reset_index(drop=True)
            pseudo_train_df = {'test_df': test_df.copy()} if 'test_df' in locals() else None
            if round_idx == 0:
                if hasattr(config, 'BEST_HP_LSTM'): delattr(config, 'BEST_HP_LSTM')
                if hasattr(config, 'BEST_HP_RNN'): delattr(config, 'BEST_HP_RNN')
                if hasattr(config, 'BEST_HP_MLP'): delattr(config, 'BEST_HP_MLP')
                run_full_workflow(base_train_df, test_df, config)
                best_hp_lstm = getattr(config, 'BEST_HP_LSTM', None)
                best_hp_rnn = getattr(config, 'BEST_HP_RNN', None)
                best_hp_mlp = getattr(config, 'BEST_HP_MLP', None)
            else:
                config.BEST_HP_LSTM = best_hp_lstm
                config.BEST_HP_RNN = best_hp_rnn
                config.BEST_HP_MLP = best_hp_mlp
                run_full_workflow(base_train_df, test_df, config)
        except Exception as e:
            pass
        finally:
            try:
                with open('results_predictions.json', 'r') as f:
                    results = json.load(f)
                ff_pred = np.array(results['first_five_pred_numbers'])
                s_pred = np.array(results['sixth_pred_number'])
                rounds_first_five.append(ff_pred)
                rounds_sixth.append(s_pred)
                # Diagnostics: print per-class prediction and true counts
                ff_pred_flat = ff_pred.flatten()
                s_pred_flat = s_pred.flatten()
                ff_true_flat = np.array(y_true_first_five).flatten()
                s_true_flat = np.array(y_true_sixth).flatten()
                ff_pred_counts = dict(zip(*np.unique(ff_pred_flat, return_counts=True)))
                s_pred_counts = dict(zip(*np.unique(s_pred_flat, return_counts=True)))
                ff_true_counts = dict(zip(*np.unique(ff_true_flat, return_counts=True)))
                s_true_counts = dict(zip(*np.unique(s_true_flat, return_counts=True)))
            except Exception as e:
                pass
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
    if len(set(valid_first)) > 1 or len(set(valid_sixth)) > 1:
        print(f"[ERROR][ensemble_predict] Prediction shape mismatch! first_five shapes: {shapes_first}, sixth shapes: {shapes_sixth}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch! See diagnostic output above.")
    strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'average').lower()
    if any(pf is None for pf in preds_first) or any(ps is None for ps in preds_sixth):
        print("[ERROR][ensemble_predict] At least one prediction is None! Indices:", [i for i,pf in enumerate(preds_first) if pf is None], [i for i,ps in enumerate(preds_sixth) if ps is None])
        raise RuntimeError("[ensemble_predict] At least one prediction is None! See diagnostic output above.")
    valid_first_shapes = [pf.shape for pf in preds_first]
    valid_sixth_shapes = [ps.shape for ps in preds_sixth]
    if len(set(valid_first_shapes)) > 1 or len(set(valid_sixth_shapes)) > 1:
        print(f"[ERROR][ensemble_predict] Prediction shape mismatch before aggregation! first_five shapes: {valid_first_shapes}, sixth shapes: {valid_sixth_shapes}")
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

import numpy as np
import keras_tuner as kt
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from models.mlp_model import MLPModel
from models.lgbm_model import LightGBMModel
from util.metrics import smooth_labels, mix_uniform, kl_to_uniform, kl_divergence
from util.plot_utils import plot_ball_distributions, plot_powerball_distribution
from data.data_handler import DataHandler
from util.plot_utils_std import plot_multi_round_true_std, plot_multi_round_pred_std, plot_multi_round_kl_divergence
import scipy.special
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
import json
import config

def run_full_workflow(train_df, test_df, config):
    """
    Full workflow: trains, tunes, evaluates, and plots.
    - Meta-parameter optimization: PSO or Bayesian (config.META_OPT_METHOD)
    - Cross-validation: k-fold if config.CV_FOLDS > 1
    - Ensembling: average, weighted, or stacking (config.ENSEMBLE_STRATEGY)
    Saves predictions and metrics for diagnostics.
    """
    look_back_window = config.LOOK_BACK_WINDOW
    X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
    X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
    if X_train.size == 0 or X_test.size == 0:
        return
    input_shape = (X_train.shape[1] // 6, 6)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
    tuner_lstm = kt.RandomSearch(
        lambda hp: LSTMModel.build_lstm_model(hp, input_shape),
        objective='val_loss',
        max_trials=config.TUNER_MAX_TRIALS,
        executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIRECTORY,
        project_name=config.TUNER_PROJECT_NAME+'_lstm'
    )
    try:
        tuner_lstm.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp_lstm = tuner_lstm.get_best_hyperparameters(1)[0]
    except Exception as e:
        return

    tuner_rnn = kt.RandomSearch(
        lambda hp: RNNModel.build_rnn_model(hp, input_shape),
        objective='val_loss',
        max_trials=config.TUNER_MAX_TRIALS,
        executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIRECTORY,
        project_name=config.TUNER_PROJECT_NAME+'_rnn'
    )
    try:
        tuner_rnn.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp_rnn = tuner_rnn.get_best_hyperparameters(1)[0]
    except Exception as e:
        return
    y_train_smoothed = [smooth_labels(y_train[0], config.LABEL_SMOOTHING), smooth_labels(y_train[1], config.LABEL_SMOOTHING)]
    y_train_smoothed = [mix_uniform(y_train_smoothed[0], config.UNIFORM_MIX_PROB), mix_uniform(y_train_smoothed[1], config.UNIFORM_MIX_PROB)]
    best_lstm_model = LSTMModel.build_lstm_model(
        best_hp_lstm, input_shape,
        use_custom_loss=True,
        force_low_units=config.FORCE_LOW_UNITS,
        force_simple=config.FORCE_SIMPLE
    )

    best_rnn_model = RNNModel.build_rnn_model(
        best_hp_rnn, input_shape
    )

    best_mlp_model = MLPModel.build_mlp_model(
        input_shape,
        num_first=5,
        num_first_classes=69,
        num_sixth_classes=26,
        hidden_units=128,
        dropout_rate=0.2
    )

    X_train_flat = X_train_reshaped.reshape(X_train_reshaped.shape[0], -1)
    X_test_flat = X_test_reshaped.reshape(X_test_reshaped.shape[0], -1)
    feature_names = list(train_df.columns) if hasattr(train_df, 'columns') else None
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 69,  # for first five balls, will be overridden for sixth
        'metric': 'multi_logloss',
        'verbosity': -1,
        'num_leaves': int(getattr(config, 'LGBM_NUM_LEAVES', 31)),
        'learning_rate': float(getattr(config, 'LGBM_LEARNING_RATE', 0.1)),
        'max_depth': int(getattr(config, 'LGBM_MAX_DEPTH', 7)),
    }
    lgbm_models_first, lgbm_model_sixth = LightGBMModel.build_lgbm_models(
        num_first=5, num_first_classes=69, num_sixth_classes=26, params=lgbm_params
    )
    try:
        LightGBMModel.fit(lgbm_models_first, lgbm_model_sixth, X_train_flat, y_train_smoothed)
    except Exception as e:
        print(f"[ERROR] Exception during LightGBM model training: {e}")
        return

    # Get predictions from LightGBM
    try:
        lgbm_first_pred, lgbm_sixth_pred = LightGBMModel.predict_proba(lgbm_models_first, lgbm_model_sixth, X_test_flat, feature_names=feature_names)
    except Exception as e:
        print("[ERROR][LGBM] Exception in LightGBMModel.predict_proba:", e)
        raise
    # Wrap LightGBM as a model-like object for ensemble_predict
    class LGBMWrap:
        def __init__(self, first, sixth):
            self.first = first
            self.sixth = sixth
        def predict(self, X, verbose=0):
            # X is (samples, time, features), flatten for LGBM
            X_flat = X.reshape(X.shape[0], -1)
            feature_names = list(train_df.columns) if hasattr(train_df, 'columns') else None
            first, sixth = LightGBMModel.predict_proba(lgbm_models_first, lgbm_model_sixth, X_flat, feature_names=feature_names)
            return first, sixth
    lgbm_wrapper = LGBMWrap(lgbm_models_first, lgbm_model_sixth)
    # --- Only LSTM model for test if config.ONLY_LSTM_MODEL is set ---
    if getattr(config, 'ONLY_LSTM_MODEL', False):
        models = [best_lstm_model]
    else:
        models = [best_lstm_model, best_rnn_model, best_mlp_model, lgbm_wrapper]
    import util.model_utils
    first_five_pred, sixth_pred = util.model_utils.ensemble_predict(models, X_test_reshaped)
    def apply_temperature_softmax(probs, temperature):
        logits = np.log(np.clip(probs, 1e-12, 1.0))
        logits /= temperature
        orig_shape = logits.shape
        logits_flat = logits.reshape(-1, logits.shape[-1])
        softmax_flat = scipy.special.softmax(logits_flat, axis=-1)
        return softmax_flat.reshape(orig_shape)
    best_temp = 1.0
    best_kl_uniform = float('inf')
    best_entropy = -float('inf')
    for temp in np.arange(config.TEMP_MIN, config.TEMP_MAX + config.TEMP_STEP, config.TEMP_STEP):
        first_five_pred_temp = apply_temperature_softmax(first_five_pred, temp)
        sixth_pred_temp = apply_temperature_softmax(sixth_pred, temp)
        kl_uniforms = []
        entropies = []
        for i in range(5):
            kl_uniforms.append(kl_to_uniform(first_five_pred_temp[:, i, :]))
            entropies.append(np.mean(entropy(first_five_pred_temp[:, i, :].T)))
        kl_uniforms.append(kl_to_uniform(sixth_pred_temp[:, 0, :]))
        entropies.append(np.mean(entropy(sixth_pred_temp[:, 0, :].T)))
        mean_kl_uniform = np.mean(kl_uniforms)
        mean_entropy = np.mean(entropies)
        if mean_kl_uniform < best_kl_uniform:
            best_kl_uniform = mean_kl_uniform
            best_entropy = mean_entropy
            best_temp = temp
    print(f"\nBest temperature found by grid search (min KL to uniform): {best_temp}")
    print(f"KL to uniform at best temperature: {best_kl_uniform:.6f}")
    print(f"Entropy at best temperature: {best_entropy:.6f}")
    # Final predictions
    first_five_pred_temp = apply_temperature_softmax(first_five_pred, best_temp)
    sixth_pred_temp = apply_temperature_softmax(sixth_pred, best_temp)

    from util.optimal_assignment import optimal_assignment
    def enforce_unique_predictions(probs):
        # Greedy uniqueness (legacy)
        num_samples, num_balls, num_classes = probs.shape
        unique_preds = np.zeros((num_samples, num_balls), dtype=int)
        for i in range(num_samples):
            chosen = set()
            for b in range(num_balls):
                sorted_idx = np.argsort(probs[i, b])[::-1]
                for idx in sorted_idx:
                    if idx not in chosen:
                        unique_preds[i, b] = idx
                        chosen.add(idx)
                        break
        return unique_preds + 1

    def optimal_assignment_predictions(probs):
        # Use Hungarian algorithm for optimal assignment
        num_samples, num_balls, num_classes = probs.shape
        assigned = np.zeros((num_samples, num_balls), dtype=int)
        for i in range(num_samples):
            assigned[i] = optimal_assignment(probs[i])
        return assigned + 1

    # Choose assignment method based on config
    assignment_method = getattr(config, 'ASSIGNMENT_METHOD', 'optimal')
    if assignment_method == 'optimal':
        first_five_pred_numbers = optimal_assignment_predictions(first_five_pred_temp)
    else:
        first_five_pred_numbers = enforce_unique_predictions(first_five_pred_temp)
    # Log assignment method
    import util.experiment_tracker
    if hasattr(util, 'experiment_tracker'):
        tracker = getattr(util, 'experiment_tracker', None)
        if tracker:
            tracker.log_metric('assignment_method', assignment_method)
    sixth_pred_number = np.argmax(sixth_pred_temp, axis=-1) + 1
    # Save softmax probabilities for pseudo-labeling filtering
    first_five_pred_softmax = first_five_pred_temp.tolist()
    sixth_pred_softmax = sixth_pred_temp.tolist()
    # True values
    y_first_five_true_numbers = np.argmax(y_test[0], axis=-1) + 1
    y_sixth_true_number = np.argmax(y_test[1], axis=-1) + 1
    # Evaluation
    num_samples = first_five_pred_numbers.shape[0]
    first_five_matches = (first_five_pred_numbers == y_first_five_true_numbers)
    sixth_matches = (sixth_pred_number == y_sixth_true_number)
    num_first_five_matched = np.sum(first_five_matches, axis=1)
    at_least_one_first_five = np.mean(num_first_five_matched >= 1)
    all_first_five = np.mean(num_first_five_matched == 5)
    powerball_match = np.mean(sixth_matches)
    all_six_match = np.mean((num_first_five_matched == 5) & (sixth_matches[:, 0]))
    print(f"At least one of first five matched: {at_least_one_first_five:.4f}")
    print(f"All first five matched: {all_first_five:.6f}")
    print(f"Powerball matched: {powerball_match:.4f}")
    print(f"All six matched: {all_six_match:.8f}")

    # Top-n accuracy metric for each ball
    top_n = 5  # You can adjust this value as needed
    print(f"Top-{top_n} accuracy per ball:")
    for i in range(5):
        probs = first_five_pred_temp[:, i, :]
        true_nums = y_first_five_true_numbers[:, i] - 1  # zero-based
        topn = np.argsort(probs, axis=1)[:, -top_n:]
        in_topn = [true_nums[j] in topn[j] for j in range(num_samples)]
        acc = np.mean(in_topn)
        print(f"  Ball {i+1}: {acc:.4f}")
    # For sixth ball
    probs6 = sixth_pred_temp[:, 0, :]
    true6 = y_sixth_true_number[:, 0] - 1
    topn6 = np.argsort(probs6, axis=1)[:, -top_n:]
    in_topn6 = [true6[j] in topn6[j] for j in range(num_samples)]
    acc6 = np.mean(in_topn6)
    print(f"  Powerball (6th): {acc6:.4f}")
    # Std & KL
    for i in range(5):
        true_std = np.std(y_first_five_true_numbers[:, i])
        pred_std = np.std(first_five_pred_numbers[:, i])
        true_hist = np.bincount(y_first_five_true_numbers[:, i]-1, minlength=69)
        pred_hist = np.bincount(first_five_pred_numbers[:, i]-1, minlength=69)
        true_dist = true_hist / np.sum(true_hist)
        pred_dist = pred_hist / np.sum(pred_hist)
        kl = kl_divergence(true_dist, pred_dist)
        print(f"Ball {i+1} True Std: {true_std:.2f}, Predicted Std: {pred_std:.2f}, KL Divergence: {kl:.6f}")
    true_std_6 = np.std(y_sixth_true_number[:, 0])
    pred_std_6 = np.std(sixth_pred_number[:, 0])
    true_hist_6 = np.bincount(y_sixth_true_number[:, 0]-1, minlength=26)
    pred_hist_6 = np.bincount(sixth_pred_number[:, 0]-1, minlength=26)
    true_dist_6 = true_hist_6 / np.sum(true_hist_6)
    pred_dist_6 = pred_hist_6 / np.sum(pred_hist_6)
    kl_6 = kl_divergence(true_dist_6, pred_dist_6)
    print(f"Powerball (6th Ball) True Std: {true_std_6:.2f}, Predicted Std: {pred_std_6:.2f}, KL Divergence: {kl_6:.6f}")

    # Save predictions and softmax for iterative stacking and pseudo-labeling
    import json
    results_to_save = {
        'first_five_pred_numbers': first_five_pred_numbers.tolist(),
        'sixth_pred_number': sixth_pred_number.tolist(),
        'first_five_pred_softmax': first_five_pred_softmax,
        'sixth_pred_softmax': sixth_pred_softmax
    }
    with open('results_predictions.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
