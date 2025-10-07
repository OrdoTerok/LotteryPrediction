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
    plot_multi_round_kl_divergence,
    plot_multi_round_true_pred_std
)
from pipeline.experiment_tracker import ExperimentTracker
from core.cache import Cache
from core.log_utils import get_logger
from optimization.meta_search import MetaParameterSearch
from core.model_utils import get_results_history

def run_pipeline(config, from_iterative_stacking=False, cv=None, best_pred=None):
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
    # Now split train/test after final_df is set
    train_df, test_df = split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    # --- Final Feature Count/Order Check Before Data Preparation ---
    print(f"[FeatureSync][PREP] train_df columns: {list(train_df.columns)}")
    print(f"[FeatureSync][PREP] test_df columns: {list(test_df.columns)}")
    assert list(train_df.columns) == list(test_df.columns), "Train/test columns do not match before data preparation!"
    # --- Feature Count/Order Checks ---
    def check_feature_consistency(X_train, X_test, logger):
        if X_train.shape[2] != X_test.shape[2]:
            logger.error(f"[Pipeline][FeatureSync] Feature count mismatch: train has {X_train.shape[2]}, test has {X_test.shape[2]}. Halting pipeline.")
            raise ValueError(f"Feature count mismatch: train has {X_train.shape[2]}, test has {X_test.shape[2]}")
    # After prepare_data_for_lstm, check feature consistency
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
    # Always split train/test after final_df is set
    train_df, test_df = split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    # --- Final Feature Count/Order Check Before Data Preparation ---
    print(f"[FeatureSync][PREP] train_df columns: {list(train_df.columns)}")
    print(f"[FeatureSync][PREP] test_df columns: {list(test_df.columns)}")
    assert list(train_df.columns) == list(test_df.columns), "Train/test columns do not match before data preparation!"
    # --- Feature Synchronization ---
    # Centralize meta_cols logic
    meta_cols = [col for col in final_df.columns if col.startswith('prev_pred_ball_') or col == 'prev_pred_sixth' or col == 'is_pseudo']
    import numpy as np
    # Get all columns that should be present in both train and test
    all_cols = sorted(set(train_df.columns).union(set(test_df.columns)).union(set(meta_cols)))
    # Add missing columns as zeros to both train and test, and reorder identically
    for df_name, df in zip(['train', 'test'], [train_df, test_df]):
        missing = [col for col in all_cols if col not in df.columns]
        if missing:
            print(f"[FeatureSync] {df_name}_df missing columns: {missing}")
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[all_cols]
        if df_name == 'train':
            train_df = df
        else:
            test_df = df
    print(f"[FeatureSync] train_df columns: {list(train_df.columns)}")
    print(f"[FeatureSync] test_df columns: {list(test_df.columns)}")
    assert list(train_df.columns) == list(test_df.columns), "Train/test columns do not match after synchronization!"
    from data import augmentation
    # Incorporate best prediction from history as pseudo-label if available
    history = get_results_history()
    if history and isinstance(history, list) and len(history) > 0:
        best_entry = history[-1]
        # Only add if not already in train_df (avoid duplicates)
        if (
            best_entry.get('first_five') is not None and 
            best_entry.get('sixth') is not None and 
            isinstance(best_entry.get('first_five'), list)
        ):
            # Create a DataFrame row for the pseudo-labeled sample
            pseudo_row = {}
            for i, val in enumerate(best_entry['first_five']):
                pseudo_row[f'ball_{i+1}'] = val
            pseudo_row['powerball'] = best_entry['sixth'][0] if isinstance(best_entry['sixth'], list) else best_entry['sixth']
            pseudo_row['is_pseudo'] = 1
            import pandas as pd
            pseudo_df = pd.DataFrame([pseudo_row])
            pseudo_cols = [f'ball_{i+1}' for i in range(5)] + ['powerball']
            missing_cols = [col for col in pseudo_cols if col not in train_df.columns]
            if missing_cols:
                logger.warning(f"[Pipeline] Skipping duplicate pseudo-label check: columns missing in train_df: {missing_cols}. Adding pseudo-labeled row without duplicate check.")
                train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
                logger.info("Added best prediction from history as pseudo-labeled sample to training data.")
            else:
                # Only add if not already present
                if not ((train_df[pseudo_cols] == pseudo_df[pseudo_cols].iloc[0]).all(axis=1)).any():
                    train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
                    logger.info("Added best prediction from history as pseudo-labeled sample to training data.")
        else:
            logger.warning("[Pipeline] best_entry['first_five'] is not a list or is missing, skipping pseudo-labeling from history.")
    if getattr(config, 'USE_PSEUDO_LABELING', False):
        logger.info("[Augmentation] Applying pseudo-labeling to training data...")
        teacher_model_type = 'lstm'  # or make this configurable
        from models.model_factory import get_model
        look_back_window = config.LOOK_BACK_WINDOW
        # Build meta_cols from union of all possible meta features in train/test
        meta_cols_sync = [col for col in train_df.columns if col.startswith('prev_pred_ball_') or col == 'prev_pred_sixth' or col == 'is_pseudo']
        # Guarantee identical meta_cols for both splits
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window, meta_cols=meta_cols_sync)
        teacher_model = get_model(teacher_model_type, input_shape=X_train.shape[1:])
        teacher_model.fit(X_train, y_train, epochs=config.EPOCHS_FINAL, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=0)
        train_df = augmentation.pseudo_label(
            train_df,
            teacher_model,
            threshold=getattr(config, 'PSEUDO_CONFIDENCE_THRESHOLD', 0.9)
        )
    if getattr(config, 'USE_NOISE_INJECTION', False):
        logger.info("[Augmentation] Applying noise injection to training features...")
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window, meta_cols=meta_cols)
        X_train = augmentation.add_gaussian_noise(
            X_train,
            std=getattr(config, 'NOISE_STD', 0.1),
            random_state=getattr(config, 'NOISE_RANDOM_STATE', None)
        )
    else:
        look_back_window = config.LOOK_BACK_WINDOW
        # Build meta_cols_sync from union of all possible meta features in train/test
        meta_cols_sync = [col for col in train_df.columns if col.startswith('prev_pred_ball_') or col == 'prev_pred_sixth' or col == 'is_pseudo']
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window, meta_cols=meta_cols_sync)

    # --- Robust Feature Synchronization (after all train_df modifications) ---
    all_cols_sync = sorted(set(train_df.columns).union(set(test_df.columns)))
    for df_name, df in zip(['train', 'test'], [train_df, test_df]):
        missing = [col for col in all_cols_sync if col not in df.columns]
        if missing:
            print(f"[FeatureSync][FINAL] {df_name}_df missing columns: {missing}")
        for col in all_cols_sync:
            if col not in df.columns:
                df[col] = 0.0
        df = df[all_cols_sync]
        if df_name == 'train':
            train_df = df
        else:
            test_df = df
    print(f"[FeatureSync][FINAL] train_df columns: {list(train_df.columns)}")
    print(f"[FeatureSync][FINAL] test_df columns: {list(test_df.columns)}")
    assert list(train_df.columns) == list(test_df.columns), "Train/test columns do not match after robust synchronization!"

    look_back_window = config.LOOK_BACK_WINDOW
    # Guarantee meta_cols_sync is defined for test extraction
    X_test, y_test = prepare_data_for_lstm(test_df, look_back=look_back_window, meta_cols=meta_cols_sync)
    print(f"[FeatureSync][EXTRACT] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    check_feature_consistency(X_train, X_test, logger)
    if X_test.size == 0:
        logger.error("Not enough data to create test sequences. Exiting.")
        tracker.end_run()
        return float('inf')
    # Check y_test structure
    if not (isinstance(y_test, (list, tuple)) and len(y_test) >= 2):
        logger.error(f"y_test is not a tuple/list with at least two elements. Got type: {type(y_test)}, value: {y_test}")
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
    cv_fold_preds_first_five = []
    cv_fold_preds_sixth = []
    cv_fold_labels = []
    per_fold_models = []  # List of lists: per fold, list of models (one per type)
    # Per-fold model storage for ensembling
    import numpy as np
    def log_shape_info(arr, name, model_type=None):
        prefix = f"[{model_type.upper()}]" if model_type else ""
        if isinstance(arr, (np.ndarray,)):
            logger.info(f"[Pipeline]{prefix}[{name}] shape: {arr.shape}, dtype: {arr.dtype}")
        elif hasattr(arr, 'shape'):
            logger.info(f"[Pipeline]{prefix}[{name}] shape: {arr.shape}")
        else:
            logger.info(f"[Pipeline]{prefix}[{name}] type: {type(arr)}")
    n_folds = getattr(config, 'CV_FOLDS', 5)
    fold_models_by_type = {mt: [] for mt in model_types}
    fold_val_idx = []  # Store validation indices for each fold
    # First, run cross-validation for each model type and store per-fold models
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_val_idx = []
    cv_fold_preds_first_five = []
    cv_fold_preds_sixth = []
    cv_fold_labels = []
    fold_models_by_type = {mt: [] for mt in model_types}
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        fold_preds_first = []
        fold_preds_sixth = []
        fold_labels = []
        for model_type in model_types:
            logger.info(f"[Pipeline] Running model: {model_type.upper()} fold {fold_idx+1}/{n_folds}")
            try:
                if model_type == 'lgbm':
                    model = get_model(model_type)
                    X_train_lgbm, y_train_lgbm = prepare_data_for_lstm(train_df, look_back=look_back_window, meta_cols=meta_cols_sync)
                    X_test_lgbm, y_test_lgbm = prepare_data_for_lstm(test_df, look_back=look_back_window, meta_cols=meta_cols_sync)
                    # Always flatten train and test the same way
                    def flatten_X(X):
                        return X.reshape((X.shape[0], -1)) if X.ndim == 3 else X
                    X_train_lgbm_flat = flatten_X(X_train_lgbm)
                    X_test_lgbm_flat = flatten_X(X_test_lgbm)
                    X_tr = X_train_lgbm_flat[train_idx]
                    X_val = X_train_lgbm_flat[val_idx]
                    # If y_train_lgbm is a tuple/list, index each element separately
                    if isinstance(y_train_lgbm, (tuple, list)):
                        y_tr = tuple(y[train_idx] for y in y_train_lgbm)
                        y_val = tuple(y[val_idx] for y in y_train_lgbm)
                    else:
                        y_tr = y_train_lgbm[train_idx]
                        y_val = y_train_lgbm[val_idx]
                    feature_names = [f"feat_{i}" for i in range(X_tr.shape[1])]
                    # Log feature counts and (if possible) column names for diagnostics
                    logger.info(f"[LGBM][CV] X_tr shape: {X_tr.shape}, X_val shape: {X_val.shape}, X_test_lgbm_flat shape: {X_test_lgbm_flat.shape}")
                    if hasattr(X_tr, 'columns'):
                        logger.info(f"[LGBM][CV] X_tr columns: {list(X_tr.columns)}")
                    if hasattr(X_test_lgbm_flat, 'columns'):
                        logger.info(f"[LGBM][CV] X_test_lgbm_flat columns: {list(X_test_lgbm_flat.columns)}")
                else:
                    X_train_seq, y_train_seq = prepare_data_for_lstm(train_df, look_back=look_back_window)
                    X_tr = X_train_seq[train_idx]
                    X_val = X_train_seq[val_idx]
                    # If y_train_seq is a tuple/list, index each element separately (applies to all non-LGBM models)
                    if isinstance(y_train_seq, (tuple, list)):
                        y_tr = tuple(y[train_idx] for y in y_train_seq)
                        y_val = tuple(y[val_idx] for y in y_train_seq)
                    else:
                        y_tr = y_train_seq[train_idx]
                        y_val = y_train_seq[val_idx]
                    input_shape = X_tr.shape[1:]
                    model = get_model(model_type, input_shape=input_shape)
                def log_shape_info(arr, name):
                    import numpy as np
                    if isinstance(arr, (np.ndarray,)):
                        logger.info(f"[Pipeline][{model_type.upper()}][{name}] shape: {arr.shape}, dtype: {arr.dtype}")
                    elif hasattr(arr, 'shape'):
                        logger.info(f"[Pipeline][{model_type.upper()}][{name}] shape: {arr.shape}")
                    else:
                        logger.info(f"[Pipeline][{model_type.upper()}][{name}] type: {type(arr)}")
                log_shape_info(X_tr, 'X_tr')
                log_shape_info(y_tr, 'y_tr')
                if model_type == 'lgbm':
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                else:
                    model.fit(X_tr, y_tr, epochs=config.EPOCHS_FINAL, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=0)
                    if hasattr(model, 'model'):
                        if X_val.shape[1:] != model.model.input_shape[1:]:
                            logger.error(f"[Pipeline] Validation data shape {X_val.shape[1:]} does not match model input shape {model.model.input_shape[1:]}. Skipping prediction.")
                            continue
                        preds = model.model.predict(X_val, verbose=0)
                    else:
                        preds = model.predict(X_val)
                if isinstance(preds, (list, tuple)):
                    pred_first = np.argmax(preds[0], axis=-1) + 1
                    pred_sixth = np.argmax(preds[1], axis=-1) + 1
                else:
                    pred_first = np.argmax(preds, axis=-1) + 1
                    pred_sixth = None
                fold_preds_first.append(pred_first)
                fold_preds_sixth.append(pred_sixth)
                fold_labels.append(f"{model_type.upper()} CV{fold_idx+1}")
                fold_models_by_type[model_type].append(model)
                logger.info(f"[Diagnostics][CV-FOLD-SUCCESS] model_type={model_type}, fold={fold_idx+1}")
            except Exception as fold_exc:
                logger.error(f"[Diagnostics][CV-FOLD-ERROR] model_type={model_type}, fold={fold_idx+1}, error={fold_exc}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        # Store predictions for plotting (not ensembled yet)
        cv_fold_preds_first_five.append(fold_preds_first)
        cv_fold_preds_sixth.append(fold_preds_sixth)
        cv_fold_labels.append(fold_labels)
        if len(fold_val_idx) < n_folds:
            fold_val_idx.append(val_idx)
            # Fit final model on all data
            log_shape_info(X_train, 'X_train')
            log_shape_info(y_train, 'y_train')
            eval_result = None
            if model_type == 'lgbm':
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, epochs=config.EPOCHS_FINAL, batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT, verbose=0)
                if X_test.shape[1:] != model.model.input_shape[1:]:
                    logger.error(f"[Pipeline] Test data shape {X_test.shape[1:]} does not match model input shape {model.model.input_shape[1:]}. Skipping evaluation.")
                else:
                    eval_result = model.evaluate(X_test, y_test, verbose=0)
            log_shape_info(X_test, 'X_test')
            log_shape_info(y_test, 'y_test')
            results[model_type] = eval_result
            trained_models.append(model)
            # Save model predictions for best-match selection
            if model_type == 'lgbm':
                # Only predict if feature count matches
                # Ensure X_test_lgbm_flat is defined
                if 'X_test_lgbm_flat' not in locals():
                    if X_test_lgbm.ndim == 3:
                        X_test_lgbm_flat = X_test_lgbm.reshape(X_test_lgbm.shape[0], -1)
                    else:
                        X_test_lgbm_flat = X_test_lgbm
                if X_test_lgbm_flat.shape[1] == X_train.shape[1]:
                    preds = model.predict(X_test_lgbm_flat, feature_names=feature_names)
                else:
                    logger.error(f"[LGBM][ERROR] Feature count mismatch: model expects {X_train.shape[1]}, but input has {X_test_lgbm_flat.shape[1]} features. Skipping prediction.")
                    preds = None
                if preds is not None:
                    if isinstance(preds, (list, tuple)) and len(preds) == 2:
                        pred_first = preds[0] + 1
                        pred_sixth = preds[1] + 1
                    else:
                        pred_first = preds + 1
                        pred_sixth = None
                else:
                    pred_first = None
                    pred_sixth = None
            else:
                # Check test data shape before prediction
                if hasattr(model, 'model') and X_test.shape[1:] != model.model.input_shape[1:]:
                    logger.error(f"[Pipeline] Test data shape {X_test.shape[1:]} does not match model input shape {model.model.input_shape[1:]}. Skipping prediction.")
                    pred_first = None
                    pred_sixth = None
                else:
                    pred_first = np.argmax(model.model.predict(X_test, verbose=0)[0], axis=-1) + 1
                    pred_sixth = np.argmax(model.model.predict(X_test, verbose=0)[1], axis=-1) + 1
            all_predictions.append({
                'source': model_type,
                'first_five': pred_first,
                'sixth': pred_sixth,
                'metrics': eval_result
            })
            logger.info(f"[Pipeline] {model_type.upper()} evaluation result: {eval_result}")
        # Exception handling is now managed inside the CV fold loop above
    logger.info(f"[Pipeline] All model results: {results}")

    # Per-fold ensembling: for each fold, ensemble the models trained in that fold and predict on the fold's validation set
    per_fold_ensemble_preds_first = []
    per_fold_ensemble_preds_sixth = []
    for fold_idx in range(n_folds):
        fold_models = [fold_models_by_type[mt][fold_idx] for mt in model_types if len(fold_models_by_type[mt]) == n_folds]
        if not fold_models:
            continue
        val_idx = fold_val_idx[fold_idx]
        X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_val = X_train[val_idx]
        from ensemble.ensemble_predict import ensemble_predict
        # Check all models can predict
        can_predict = True
        for i, m in enumerate(fold_models):
            try:
                if m.__class__.__name__ == 'LightGBMModel':
                    # Skip single-sample prediction check for LightGBMModel
                    continue
                if hasattr(m, 'model') and hasattr(m.model, 'predict'):
                    _ = m.model.predict(X_val[:1], verbose=0)
                elif hasattr(m, 'predict'):
                    _ = m.predict(X_val[:1])
                else:
                    raise AttributeError(f"Model {type(m)} has no predict method.")
            except Exception as e:
                logger.warning(f"[Per-Fold Ensemble] Model {i} ({type(m)}) failed to predict: {e}")
                can_predict = False
        if not can_predict:
            logger.warning(f"[Per-Fold Ensemble] Skipping fold {fold_idx+1} due to model prediction failure.")
            continue
        try:
            ensemble_first, ensemble_sixth = ensemble_predict(fold_models, X_val, config)
            pred_first = np.argmax(ensemble_first, axis=-1) + 1
            pred_sixth = np.argmax(ensemble_sixth, axis=-1) + 1
            per_fold_ensemble_preds_first.append(pred_first)
            per_fold_ensemble_preds_sixth.append(pred_sixth)
            cv_fold_labels.append(f"Ensemble CV{fold_idx+1}")
        except Exception as e:
            logger.warning(f"[Per-Fold Ensemble] Skipping fold {fold_idx+1} due to ensemble_predict error: {e}")
            continue

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
    if 'ensemble_first' in locals() and 'ensemble_sixth' in locals():
        from ensemble.calibration import TemperatureScaler, PlattScaler, IsotonicCalibrator
        calibration_method = getattr(config, 'CALIBRATION_METHOD', 'none').lower()
        def log_shape_type(var, name):
            logger.info(f"[Calibration] {name}: type={type(var)}, shape={getattr(var, 'shape', None)}")
        log_shape_type(ensemble_first, 'ensemble_first')
        log_shape_type(ensemble_sixth, 'ensemble_sixth')
        log_shape_type(y_test, 'y_test')
        if calibration_method != 'none':
            try:
                # Check shapes before calibration
                if not (hasattr(ensemble_first, 'shape') and len(ensemble_first.shape) >= 2):
                    logger.warning(f"[Calibration] ensemble_first shape invalid: {getattr(ensemble_first, 'shape', None)}. Skipping calibration.")
                elif not (isinstance(y_test, (list, tuple)) and len(y_test) >= 1 and hasattr(y_test[0], 'shape')):
                    logger.warning(f"[Calibration] y_test[0] shape invalid: {getattr(y_test[0], 'shape', None) if isinstance(y_test, (list, tuple)) and len(y_test) > 0 else None}. Skipping calibration.")
                else:
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
            except Exception as e:
                logger.error(f"[Calibration] Exception during calibration: {e}")
                logger.error(f"[Calibration] ensemble_first: {repr(ensemble_first)}")
                logger.error(f"[Calibration] y_test: {repr(y_test)}")
    else:
        logger.warning("[Pipeline] Skipping calibration: ensemble_first or ensemble_sixth not available.")


    # Compute final predictions from ensemble (class indices, 1-based) only if available
    if 'ensemble_first' in locals() and 'ensemble_sixth' in locals():
        final_pred_first_five = np.argmax(ensemble_first, axis=-1) + 1
        final_pred_sixth = np.argmax(ensemble_sixth, axis=-1) + 1
        # Add per-fold ensemble predictions to rounds for plotting
        # Use only per-fold ensemble predictions and the final ensemble for plotting
        # Only append final_pred_first_five/sixth if not None
        rounds_first_five = per_fold_ensemble_preds_first.copy()
        rounds_sixth = per_fold_ensemble_preds_sixth.copy()
        round_labels = [f"Ensemble CV{idx+1}" for idx in range(len(per_fold_ensemble_preds_first))]
        if final_pred_first_five is not None:
            rounds_first_five.append(final_pred_first_five)
            round_labels.append('Final')
        if final_pred_sixth is not None:
            rounds_sixth.append(final_pred_sixth)
    else:
        logger.warning("[Pipeline] Skipping final ensemble predictions: ensemble_first or ensemble_sixth not available.")
        final_pred_first_five = None
        final_pred_sixth = None
        rounds_first_five = per_fold_ensemble_preds_first
        rounds_sixth = per_fold_ensemble_preds_sixth
        round_labels = [f"Ensemble CV{idx+1}" for idx in range(len(per_fold_ensemble_preds_first))]

    # Select the best prediction by highest number of balls matched
    def count_matches(pred_first, pred_sixth, y_true_first_five, y_true_sixth):
        # pred_first, y_true_first_five: (n_samples, 5)
        # pred_sixth, y_true_sixth: (n_samples, 1)
        import numpy as np
        pred_first = np.array(pred_first)
        y_true_first_five = np.array(y_true_first_five)
        pred_sixth = np.array(pred_sixth)
        y_true_sixth = np.array(y_true_sixth)
        matches_first = pred_first == y_true_first_five
        matches_sixth = pred_sixth == y_true_sixth
        # If result is array, sum; if scalar, cast to int
        if hasattr(matches_first, 'sum'):
            matches_first = matches_first.sum()
        else:
            matches_first = int(matches_first)
        if hasattr(matches_sixth, 'sum'):
            matches_sixth = matches_sixth.sum()
        else:
            matches_sixth = int(matches_sixth)
        return matches_first + matches_sixth

    import time
    # Accept best_pred as argument, update if better found
    import inspect
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    best_pred = values.get('best_pred', None)
    best_score = -1 if best_pred is None else best_pred.get('matches', -1)
    # Ensure we compare indices, not one-hot/probabilities
    def to_indices(arr):
        arr = np.array(arr)
        # If one-hot/probabilities, take argmax; else, return as is
        if arr.ndim == 3:
            return np.argmax(arr, axis=-1) + 1
        return arr

    for pred in all_predictions:
        # Use indices for comparison
        pred_first_idx = to_indices(pred['first_five'])
        pred_sixth_idx = to_indices(pred['sixth'])
        if pred['source'] == 'ensemble':
            true_first_idx = to_indices(final_pred_first_five)
            true_sixth_idx = to_indices(final_pred_sixth)
        else:
            true_first_idx = to_indices(y_true_first_five)
            true_sixth_idx = to_indices(y_true_sixth)
        score = count_matches(pred_first_idx, pred_sixth_idx, true_first_idx, true_sixth_idx)
        if score > best_score:
            best_score = score
            best_pred = pred
    # Ensure only a single set of numbers is saved for first_five and sixth
    def flatten_prediction(pred, expected_len):
        if pred is None:
            return None
        arr = np.array(pred)
        # If 2D, take the first row
        if arr.ndim == 2:
            arr = arr[0]
        arr = arr.tolist()
        # If still too long, slice to expected_len
        if isinstance(arr, list) and len(arr) > expected_len:
            arr = arr[:expected_len]
        return arr

    first_five_flat = flatten_prediction(best_pred['first_five'], 5) if best_pred is not None else None
    sixth_flat = flatten_prediction(best_pred['sixth'], 1) if best_pred is not None else None
    # If sixth_flat is a list, extract the first element
    if isinstance(sixth_flat, list) and len(sixth_flat) == 1:
        sixth_flat = sixth_flat[0]
    best_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': best_pred['source'] if best_pred else None,
        'first_five': first_five_flat,
        'sixth': sixth_flat,
        'metrics': best_pred['metrics'] if best_pred is not None else {},
        'matches': best_score
    }

    # Utility to plot and log artifact
    def log_plot_and_artifact(plot_func, plot_args, artifact_path):
        import matplotlib.pyplot as plt
        plot_func(**plot_args)
        if os.path.exists(artifact_path):
            tracker.log_artifact(artifact_path)
        plt.show()

    # Prepare data for multi-round plots
    # Add CV fold predictions to rounds for plotting
    rounds_first_five = cv_fold_preds_first_five + [final_pred_first_five]
    rounds_sixth = cv_fold_preds_sixth + [final_pred_sixth]
    round_labels = cv_fold_labels + ['Final']
    # Use previous predictions if available
    def valid_prev_pred(pred):
        return pred is not None and hasattr(pred, 'ndim') and pred.ndim >= 2
    prev_pred_first_five = prev_pred_first_five if valid_prev_pred(prev_pred_first_five) else None
    prev_pred_sixth = prev_pred_sixth if valid_prev_pred(prev_pred_sixth) else None

    # Log and plot artifacts
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
        # Combine all six balls for std plot
        if y_true_first_five.shape[0] == y_true_sixth.shape[0]:
            y_true_all = np.concatenate([y_true_first_five, y_true_sixth.reshape(-1, 1)], axis=1)
        else:
            logger.warning(f"[Pipeline] Skipping y_true_all concatenation: shape mismatch (first_five: {y_true_first_five.shape}, sixth: {y_true_sixth.shape})")
            y_true_all = None
        rounds_all = []
        for idx, (r5, r6) in enumerate(zip(rounds_first_five, rounds_sixth)):
            if r6 is None or r5 is None:
                logger.warning(f"[Pipeline] Skipping round {idx+1} for std plot: missing prediction.")
                continue
            r5_arr = np.array(r5)
            r6_arr = np.array(r6).reshape(-1, 1)
            # Skip if either is not at least 1D or is empty
            if r5_arr.ndim < 1 or r6_arr.ndim < 1 or r5_arr.size == 0 or r6_arr.size == 0:
                logger.warning(f"[Pipeline] Skipping round {idx+1} for std plot: zero-dimensional or empty array.")
                continue
            if r5_arr.shape[0] == r6_arr.shape[0]:
                try:
                    rounds_all.append(np.concatenate([r5_arr, r6_arr], axis=1))
                except Exception as e:
                    logger.warning(f"[Pipeline] Skipping round {idx+1} for std plot: concatenate error: {e}")
                    continue
            else:
                logger.warning(f"[Pipeline] Skipping round {idx+1} for std plot: shape mismatch (r5: {r5_arr.shape}, r6: {r6_arr.shape})")
                continue
        prev_pred_all = None
        if prev_pred_first_five is not None and prev_pred_sixth is not None:
            min_len = min(prev_pred_first_five.shape[0], prev_pred_sixth.shape[0])
            if min_len == 0:
                logger.warning(f"[Pipeline] Skipping prev_pred_all concatenation: no common samples (first_five: {prev_pred_first_five.shape}, sixth: {prev_pred_sixth.shape})")
                prev_pred_all = None
            else:
                trimmed_first_five = prev_pred_first_five[:min_len]
                trimmed_sixth = prev_pred_sixth[:min_len].reshape(-1, 1)
                prev_pred_all = np.concatenate([trimmed_first_five, trimmed_sixth], axis=1)
        # Plot true vs predicted std for all balls side by side
        log_plot_and_artifact(
            plot_multi_round_true_pred_std,
            dict(
                y_true=y_true_all,
                pred_rounds_list=rounds_all,
                prev_true=prev_pred_all,  # previous true values (if available)
                prev_pred=prev_pred_all,  # previous pred values (if available)
                round_labels=round_labels,
                prev_label='Previous'
            ),
            'multi_round_true_pred_std.png'
        )
        log_plot_and_artifact(
            plot_multi_round_kl_divergence,
            dict(
                y_true=y_true_all,
                rounds_pred_list=rounds_all,
                prev_pred=prev_pred_all,
                num_balls=6,
                n_classes=[69, 69, 69, 69, 69, 26],
                round_labels=round_labels,
                prev_label='Previous'
            ),
            'multi_round_kl_divergence.png'
        )

    return best_entry

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

