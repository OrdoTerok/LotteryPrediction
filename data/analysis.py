"""
data.analysis
-------------
Analysis utilities for lottery datasets, including value range and frequency analysis for each ball.
Functions:
    - analyze_value_ranges_per_ball: Print and log statistics for each ball in the dataset.
"""
import numpy as np
import pandas as pd

def analyze_value_ranges_per_ball(df):
    """
    Print and log min, max, percentiles, and top frequencies for each ball in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Winning Numbers' column containing space-separated numbers.
    """
    winning_numbers = df['Winning Numbers'].str.split().apply(lambda x: [int(i) for i in x]).tolist()
    balls = np.array(winning_numbers, dtype=int)
    import logging
    logger = logging.getLogger(__name__)
    # Vectorized stats for first 5 balls
    for i in range(5):
        ball_vals = balls[:, i]
        logger.info(f"\nBall {i+1}:")
        logger.info(f"  Min: {ball_vals.min()}, Max: {ball_vals.max()}")
        logger.info(f"  25th percentile: {np.percentile(ball_vals, 25):.1f}, 75th percentile: {np.percentile(ball_vals, 75):.1f}")
        vc = pd.Series(ball_vals).value_counts().sort_index()
        top = vc.sort_values(ascending=False).head(10)
        logger.info(f"  Top 10 most frequent numbers: {list(top.index)} (counts: {list(top.values)})")
    # Powerball (6th ball)
    ball_vals = balls[:, 5]
    logger.info(f"\nPowerball (6th Ball):")
    logger.info(f"  Min: {ball_vals.min()}, Max: {ball_vals.max()}")
    logger.info(f"  25th percentile: {np.percentile(ball_vals, 25):.1f}, 75th percentile: {np.percentile(ball_vals, 75):.1f}")
    vc = pd.Series(ball_vals).value_counts().sort_index()
    top = vc.sort_values(ascending=False).head(10)
    logger.info(f"  Top 10 most frequent numbers: {list(top.index)} (counts: {list(top.values)})")
