"""
Pipeline Orchestration Module
============================
This module provides the main pipeline logic for the LotteryPrediction project, including:
    - Data loading and preprocessing
    - Meta-optimization
    - Iterative stacking
    - Evaluation and result tracking
    - Plotting and artifact management

The main entry point is `run_pipeline`, which coordinates the end-to-end workflow.
"""
import os
import json
import numpy as np
from data.loaders import fetch_data_from_datagov, load_data_from_kaggle
from data.preprocessing import combine_and_clean_data, save_to_file, prepare_data_for_lstm
from data.split import split_dataframe_by_percentage
from visualization.plot_utils import (
    plot_multi_round_ball_distributions,
    plot_multi_round_powerball_distribution
)
from visualization.plot_utils_std import (
    plot_multi_round_true_std,
    plot_multi_round_pred_std,
    plot_multi_round_kl_divergence
)
from pipeline.experiment_tracker import ExperimentTracker
from core.cache import Cache
from core.log_utils import get_logger
from optimization.meta_search import MetaParameterSearch
from util.model_utils import get_results_history

def run_pipeline(config, from_iterative_stacking=False, cv=None):
    """
    Orchestrates the full pipeline for LotteryPrediction.

    Args:
        config: Configuration object with pipeline parameters.
        from_iterative_stacking (bool): If True, called from iterative stacking context.
        cv: Optional cross-validation parameter.

    Returns:
        Depends on pipeline logic (e.g., evaluation metrics, predictions, or None).
    """
    DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'
    logger = get_logger()
    cache = Cache()
    tracker = ExperimentTracker()
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
    from data import augmentation
    if getattr(config, 'USE_PSEUDO_LABELING', False):
        logger.info("[Augmentation] Applying pseudo-labeling to training data...")
        # Use the first model type as the teacher for pseudo-labeling
        teacher_model_type = 'lstm'  # or make this configurable
        from models.model_factory import get_model
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
        teacher_model = get_model(teacher_model_type, input_shape=X_train.shape[1:])
        teacher_model.fit(X_train, y_train, epochs=config.EPOCHS_FINAL, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=0)
        # Apply pseudo-labeling
        train_df = augmentation.pseudo_label(
            train_df,
            teacher_model,
            threshold=getattr(config, 'PSEUDO_CONFIDENCE_THRESHOLD', 0.9)
        )
    if getattr(config, 'USE_NOISE_INJECTION', False):
        logger.info("[Augmentation] Applying noise injection to training features...")
        # Prepare X_train for noise injection
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_train = augmentation.add_gaussian_noise(
            X_train,
            std=getattr(config, 'NOISE_STD', 0.1),
            random_state=getattr(config, 'NOISE_RANDOM_STATE', None)
        )
        # Optionally, update train_df or pass X_train directly to model training
    else:
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
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

    # Load prediction history for reference or analysis
    history = get_results_history()
    # Optionally log or use history as needed
    # logger.info(f"Loaded {len(history)} previous prediction results.")

    # Extract previous predictions from history (most recent entry)
    if history and isinstance(history, list) and len(history) > 0:
        last_entry = history[-1]
        if isinstance(last_entry, dict):
            prev_pred_first_five = np.array(last_entry.get('first_five'))
            prev_pred_sixth = np.array(last_entry.get('sixth'))

    # Assignment method selection
    from core import optimal_assignment
    def assign_predictions(prob_matrix):
        if getattr(config, 'ASSIGNMENT_METHOD', 'optimal') == 'optimal':
            return optimal_assignment.optimal_assignment(prob_matrix)
        else:
            return np.argmax(prob_matrix, axis=-1)

    # Track all predictions for best-match selection
    all_predictions = []

    # Run all model types: LSTM, RNN, MLP, LGBM
    from models.model_factory import get_model
    model_types = ['lstm', 'rnn', 'mlp', 'lgbm']
    results = {}
    trained_models = []
    for model_type in model_types:
        logger.info(f"[Pipeline] Running model: {model_type.upper()}")
        try:
            model = get_model(model_type, input_shape=X_test.shape[1:])
            X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
            model.fit(X_train, y_train, epochs=config.EPOCHS_FINAL, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=0)
            eval_result = model.evaluate(X_test, y_test, verbose=0)
            results[model_type] = eval_result
            trained_models.append(model)
            # Save model predictions for best-match selection
            pred_first = np.argmax(model.model.predict(X_test, verbose=0)[0], axis=-1) + 1
            pred_sixth = np.argmax(model.model.predict(X_test, verbose=0)[1], axis=-1) + 1
            all_predictions.append({
                'source': model_type,
                'first_five': pred_first,
                'sixth': pred_sixth,
                'metrics': eval_result
            })
            logger.info(f"[Pipeline] {model_type.upper()} evaluation result: {eval_result}")
        except Exception as e:
            logger.error(f"[Pipeline] Error running model {model_type}: {e}")
    logger.info(f"[Pipeline] All model results: {results}")

    # Ensemble predictions from all models
    try:
        from ensemble.ensemble_predict import ensemble_predict
        ensemble_first, ensemble_sixth = ensemble_predict(trained_models, X_test, config)
        logger.info(f"[Pipeline] Ensemble predictions complete. Shapes: first_five={ensemble_first.shape}, sixth={ensemble_sixth.shape}")
        # Save ensemble predictions for best-match selection
        pred_first = np.argmax(ensemble_first, axis=-1) + 1
        pred_sixth = np.argmax(ensemble_sixth, axis=-1) + 1
        all_predictions.append({
            'source': 'ensemble',
            'first_five': pred_first,
            'sixth': pred_sixth,
            'metrics': {},
        })
    except Exception as e:
        logger.error(f"[Pipeline] Error during ensembling: {e}")

    # Calibration method selection and application
    from ensemble.calibration import TemperatureScaler, PlattScaler, IsotonicCalibrator
    calibration_method = getattr(config, 'CALIBRATION_METHOD', 'none').lower()
    if calibration_method != 'none':
        logger.info(f"[Pipeline] Applying calibration method: {calibration_method}")
        # Example: use y_test[0] and y_test[1] as labels for calibration
        if calibration_method == 'temperature':
            scaler_first = TemperatureScaler()
            scaler_first.fit(ensemble_first.reshape(-1, ensemble_first.shape[-1]), np.argmax(y_test[0], axis=-1).flatten())
            ensemble_first = scaler_first.transform(ensemble_first.reshape(-1, ensemble_first.shape[-1])).reshape(ensemble_first.shape)
            scaler_sixth = TemperatureScaler()
            scaler_sixth.fit(ensemble_sixth.reshape(-1, ensemble_sixth.shape[-1]), np.argmax(y_test[1], axis=-1).flatten())
            ensemble_sixth = scaler_sixth.transform(ensemble_sixth.reshape(-1, ensemble_sixth.shape[-1])).reshape(ensemble_sixth.shape)
        elif calibration_method == 'platt':
            scaler_first = PlattScaler()
            scaler_first.fit(ensemble_first.reshape(-1, ensemble_first.shape[-1]), np.argmax(y_test[0], axis=-1).flatten())
            ensemble_first = scaler_first.transform(ensemble_first.reshape(-1, ensemble_first.shape[-1])).reshape(ensemble_first.shape)
            scaler_sixth = PlattScaler()
            scaler_sixth.fit(ensemble_sixth.reshape(-1, ensemble_sixth.shape[-1]), np.argmax(y_test[1], axis=-1).flatten())
            ensemble_sixth = scaler_sixth.transform(ensemble_sixth.reshape(-1, ensemble_sixth.shape[-1])).reshape(ensemble_sixth.shape)
        elif calibration_method == 'isotonic':
            calibrator_first = IsotonicCalibrator()
            calibrator_first.fit(ensemble_first.reshape(-1, ensemble_first.shape[-1]), np.argmax(y_test[0], axis=-1).flatten())
            ensemble_first = calibrator_first.transform(ensemble_first.reshape(-1, ensemble_first.shape[-1])).reshape(ensemble_first.shape)
            calibrator_sixth = IsotonicCalibrator()
            calibrator_sixth.fit(ensemble_sixth.reshape(-1, ensemble_sixth.shape[-1]), np.argmax(y_test[1], axis=-1).flatten())
            ensemble_sixth = calibrator_sixth.transform(ensemble_sixth.reshape(-1, ensemble_sixth.shape[-1])).reshape(ensemble_sixth.shape)

    # Compute final predictions from ensemble (class indices, 1-based)
    final_pred_first_five = np.argmax(ensemble_first, axis=-1) + 1
    final_pred_sixth = np.argmax(ensemble_sixth, axis=-1) + 1

    # Select the best prediction by highest number of balls matched
    def count_matches(pred_first, pred_sixth, y_true_first_five, y_true_sixth):
        # pred_first, y_true_first_five: (n_samples, 5)
        # pred_sixth, y_true_sixth: (n_samples, 1)
        matches_first = (pred_first == y_true_first_five).sum()
        matches_sixth = (pred_sixth == y_true_sixth).sum()
        return matches_first + matches_sixth

    best_pred = None
    best_score = -1
    for pred in all_predictions:
        score = count_matches(
            pred['first_five'], final_pred_first_five if pred['source']=='ensemble' else y_true_first_five,  # Use y_true_first_five for all
            pred['sixth'], final_pred_sixth if pred['source']=='ensemble' else y_true_sixth
        )
        if score > best_score:
            best_score = score
            best_pred = pred

    # Save the best prediction from all runs to results_predictions_history.json
    import time
    from util.log_utils import save_json
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    best_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': best_pred['source'] if best_pred else None,
        'first_five': best_pred['first_five'].tolist() if best_pred is not None else None,
        'sixth': best_pred['sixth'].tolist() if best_pred is not None else None,
        'metrics': best_pred['metrics'] if best_pred is not None else {},
        'matches': best_score
    }
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = []
        history_data.append(best_entry)
        save_json(history_data, history_path)
        logger.info(f"Saved best prediction to {history_path}")
    except Exception as e:
        logger.error(f"Failed to save best prediction to {history_path}: {e}")

def run_meta_optimization(final_df, config):
    """
    Run meta-parameter optimization (PSO or Bayesian) and update config with best values.

    Args:
        final_df: Final combined DataFrame for training/testing.
        config: Configuration object to update with best meta-parameters.

    Returns:
        None. Updates config in-place.
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
        (0.0, 2.0),
        (7, 127),
        (0.01, 0.3),
        (3, 12)
    ]
    import data.split
    train_df, test_df = data.split.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    meta_search = MetaParameterSearch(method=getattr(config, 'META_OPT_METHOD', 'pso'))
    best = meta_search.search(
        var_names,
        bounds,
        (train_df, test_df),
        n_trials=getattr(config, 'PSO_ITER', 10),
        n_particles=getattr(config, 'PSO_PARTICLES', 5),
        n_iter=getattr(config, 'PSO_ITER', 10)
    )
    logger = get_logger()
    if best is None:
        logger.error("Meta-optimization failed or was aborted (e.g., due to recursion guard). Skipping meta-parameter update.")
        return
    logger.info(f"Best meta-hyperparameters ({getattr(config, 'META_OPT_METHOD', 'pso')}): %s", dict(zip(var_names, best)))
    for i, name in enumerate(var_names):
        setattr(config, name, best[i])

