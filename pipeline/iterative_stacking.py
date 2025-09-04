"""
Iterative Stacking Module
========================
This module implements iterative stacking logic for meta-feature integration in the LotteryPrediction pipeline.
It provides the `run_iterative_stacking` function, which performs multiple rounds of stacking and meta-feature updates.
"""
import numpy as np
from util.log_utils import get_logger
from pipeline.run_pipeline import run_pipeline

def run_iterative_stacking(train_df, test_df, config, y_true_first_five, y_true_sixth, prev_pred_first_five=None, prev_pred_sixth=None):
    """
    Run iterative stacking rounds for meta-feature integration.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        config: Configuration object with stacking parameters.
        y_true_first_five: True labels for first five balls.
        y_true_sixth: True labels for sixth ball.
        prev_pred_first_five: Previous predictions for first five balls (optional).
        prev_pred_sixth: Previous predictions for sixth ball (optional).

    Returns:
        tuple: (rounds_first_five, rounds_sixth, round_labels)
    """
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
        logger.info(f"[Iterative Stacking] Running pipeline for round {round_idx+1}...")
        # This will recursively call run_pipeline with from_iterative_stacking=True
        ensemble_first, ensemble_sixth = run_pipeline(config, from_iterative_stacking=True)
        logger.info(f"[Iterative Stacking] Pipeline complete for round {round_idx+1}. Ensemble predictions captured.")
        if ensemble_first is not None and ensemble_sixth is not None:
            rounds_first_five.append(np.asarray(ensemble_first))
            rounds_sixth.append(np.asarray(ensemble_sixth))
        else:
            logger.warning(f"[Iterative Stacking] Ensemble predictions missing for round {round_idx+1}.")
        if round_idx < num_rounds - 1 and ensemble_first is not None and ensemble_sixth is not None:
            try:
                pred_balls = np.argmax(ensemble_first, axis=-1) + 1
                pred_sixth = np.argmax(ensemble_sixth, axis=-1) + 1
                for j in range(5):
                    base_train_df[f'prev_pred_ball_{j+1}'] = pred_balls[:, j]
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
