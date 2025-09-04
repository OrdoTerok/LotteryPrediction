# data/augmentation.py
# Pseudo-labeling and noise injection utilities for data augmentation.

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def add_gaussian_noise(X, std=0.1, random_state=None):
    """
    Add Gaussian noise to features.
    Args:
        X (np.ndarray): Input features.
        std (float): Standard deviation of noise.
        random_state (int or None): Seed for reproducibility.
    Returns:
        np.ndarray: Noisy features.
    """
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, std, X.shape)
    return X + noise

def pseudo_label(df, model, threshold=0.9):
    """
    Generate pseudo-labels for unlabeled data using model predictions above a confidence threshold.
    Args:
        df (pd.DataFrame): DataFrame with features to pseudo-label.
        model: Trained model with predict_proba method.
        threshold (float): Confidence threshold for accepting pseudo-labels.
    Returns:
        pd.DataFrame: DataFrame with pseudo-labeled samples appended.
    """
    X = ... # Extract features from df as needed
    proba = model.predict_proba(X)
    max_proba = np.max(proba, axis=1)
    pseudo_labels = np.argmax(proba, axis=1)
    mask = max_proba >= threshold
    pseudo_df = df.iloc[mask].copy()
    pseudo_df['pseudo_label'] = pseudo_labels[mask]
    pseudo_df['is_pseudo'] = 1
    logger.info(f"Added {pseudo_df.shape[0]} pseudo-labeled samples.")
    return pd.concat([df, pseudo_df], ignore_index=True)
