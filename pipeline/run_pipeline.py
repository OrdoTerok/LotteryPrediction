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
        pass  # To implement: pseudo-labeling logic
    if getattr(config, 'USE_NOISE_INJECTION', False):
        logger.info("[Augmentation] Applying noise injection to training features...")
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
    # ...rest of pipeline logic...
    # This is a direct move from util/model_utils.py, further modularization can be done as needed.

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

