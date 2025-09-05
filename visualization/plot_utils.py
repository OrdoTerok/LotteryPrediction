"""
Plotting utilities for LotteryPrediction: multi-round and multi-run plots.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def plot_multi_round_powerball_distribution(y_true, rounds_pred_list, prev_pred=None, n_classes=26, title='Powerball (6th Ball) Distribution', round_labels=None, prev_label='Previous'):
    """
    Plot the distribution of true Powerball values, each round's predictions, and optionally previous predictions.
    Args:
        y_true: (n_samples, 1) true Powerball numbers
        rounds_pred_list: list of (n_samples, 1) arrays, one per round
        prev_pred: (n_samples, 1) array of previous predictions (optional)
        n_classes: number of possible values (default 26)
        title: plot title
        round_labels: list of labels for each round (optional)
        prev_label: label for previous predictions
    """
    palette = plt.get_cmap('tab10')
    plt.figure(figsize=(12, 5))
    x = np.arange(1, n_classes + 1)
    width = 0.8 / (2 + len(rounds_pred_list))
    offsets = np.linspace(-width * (1 + len(rounds_pred_list)) / 2, width * (1 + len(rounds_pred_list)) / 2, 2 + len(rounds_pred_list))
    logger.info("[PLOT DIAG] y_true (first 5): %s", y_true[:5])
    if prev_pred is not None:
        logger.info("[PLOT DIAG] prev_pred (first 5): %s", prev_pred[:5])
    # Vectorized: precompute bincounts for all rounds
    for idx, y_pred in enumerate(rounds_pred_list):
        logger.info(f"[PLOT DIAG] round {idx+1} y_pred (first 5): %s", y_pred[:5])
    true_counts = np.bincount(y_true[:, 0] - 1, minlength=n_classes)
    plt.bar(x + offsets[0], true_counts, width=width, color='blue', label='True', align='center')
    idx_offset = 1
    if prev_pred is not None:
        prev_counts = np.bincount(prev_pred[:, 0].astype(int) - 1, minlength=n_classes)
        plt.bar(x + offsets[1], prev_counts, width=width, color='black', label=prev_label, align='center')
        idx_offset += 1
    # Vectorized bincounts for all rounds
    round_counts_arr = np.stack([np.bincount(y_pred[:, 0] - 1, minlength=n_classes) for y_pred in rounds_pred_list], axis=0)
    used_labels = set()
    def make_unique(label):
        orig = label
        count = 2
        while label in used_labels:
            label = f"{orig} ({count})"
            count += 1
        used_labels.add(label)
        return label
    for idx in range(round_counts_arr.shape[0]):
        label = round_labels[idx] if round_labels and idx < len(round_labels) else f'Round {idx+1}'
        label = make_unique(label)
        color = palette((idx + 2) % 10)
        plt.bar(x + offsets[idx + idx_offset], round_counts_arr[idx], width=width, color=color, label=label, align='center')
    plt.title(title)
    plt.xlabel('Number')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_multi_round_ball_distributions(y_true, rounds_pred_list, prev_pred=None, num_balls=5, n_classes=69, title_prefix='Ball', round_labels=None, prev_label='Previous'):
    """
    Plot the distribution of true values, each round's predictions, and optionally previous predictions for each ball.
    Args:
        y_true: (n_samples, num_balls) true numbers
        rounds_pred_list: list of (n_samples, num_balls) arrays, one per round
        prev_pred: (n_samples, num_balls) array of previous predictions (optional)
        num_balls: number of balls (default 5)
        n_classes: number of possible values per ball (default 69)
        title_prefix: plot title prefix
        round_labels: list of labels for each round (optional)
        prev_label: label for previous predictions
    """
    palette = plt.get_cmap('tab10')
    for i in range(num_balls):
        logger.info(f"[PLOT DIAG] Ball {i+1} y_true (first 5): %s", y_true[:5, i])
        if prev_pred is not None:
            logger.info(f"[PLOT DIAG] Ball {i+1} prev_pred (first 5): %s", prev_pred[:5, i])
        for idx, y_pred in enumerate(rounds_pred_list):
            logger.info(f"[PLOT DIAG] Ball {i+1} round {idx+1} y_pred (first 5): %s", y_pred[:5, i])
        plt.figure(figsize=(12, 5))
        x = np.arange(1, n_classes + 1)
        width = 0.8 / (2 + len(rounds_pred_list))  # bar width
        offsets = np.linspace(-width * (1 + len(rounds_pred_list)) / 2, width * (1 + len(rounds_pred_list)) / 2, 2 + len(rounds_pred_list))
        true_counts = np.bincount(y_true[:, i] - 1, minlength=n_classes)
        plt.bar(x + offsets[0], true_counts, width=width, color='blue', label='True', align='center')
        idx_offset = 1
        if prev_pred is not None:
            prev_counts = np.bincount(prev_pred[:, i].astype(int) - 1, minlength=n_classes)
            plt.bar(x + offsets[1], prev_counts, width=width, color='black', label=prev_label, align='center')
            idx_offset += 1
        # Vectorized bincounts for all rounds for this ball
        round_counts_arr = np.stack([np.bincount(y_pred[:, i] - 1, minlength=n_classes) for y_pred in rounds_pred_list], axis=0)
        used_labels = set()
        def make_unique(label):
            orig = label
            count = 2
            while label in used_labels:
                label = f"{orig} ({count})"
                count += 1
            used_labels.add(label)
            return label
        for idx in range(round_counts_arr.shape[0]):
            label = round_labels[idx] if round_labels and idx < len(round_labels) else f'Round {idx+1}'
            label = make_unique(label)
            color = palette((idx + 2) % 10)
            plt.bar(x + offsets[idx + idx_offset], round_counts_arr[idx], width=width, color=color, label=label, align='center')
        plt.title(f'{title_prefix} {i+1} Distribution (1-{n_classes})')
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()
