
"""
data.augmentation
-----------------
Utilities for data augmentation, including pseudo-labeling and noise injection.
Functions:
    - add_gaussian_noise: Add Gaussian noise to features.
    - pseudo_label: Generate pseudo-labels for unlabeled data using model predictions.
"""

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def add_gaussian_noise(X, std=0.1, random_state=None):
    """
    Add Gaussian noise to features.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    std : float, optional
        Standard deviation of noise (default 0.1).
    random_state : int or None, optional
        Seed for reproducibility (default None).

    Returns
    -------
    np.ndarray
        Noisy features with the same shape as X.
    """
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, std, X.shape)
    return X + noise

def pseudo_label(df, model, threshold=0.9):
    """
    Generate pseudo-labels for unlabeled data using model predictions above a confidence threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features to pseudo-label.
    model : object
        Trained model with a predict_proba method.
    threshold : float, optional
        Confidence threshold for accepting pseudo-labels (default 0.9).

    Returns
    -------
    pd.DataFrame
        DataFrame with pseudo-labeled samples appended.
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
