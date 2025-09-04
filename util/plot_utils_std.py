import logging
# Remove or comment out the following block to avoid overwriting the main log file
# logging.basicConfig(
#     filename='log.rtf',
#     filemode='a',
#     format='%(asctime)s %(levelname)s: %(message)s',
#     level=logging.INFO
# )
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

def plot_multi_round_true_std(y_true, rounds_pred_list, prev_pred=None, num_balls=5, round_labels=None, prev_label='Previous'):
    """
    Plot the true standard deviation for each ball across rounds and previous runs.
    Args:
        y_true: (n_samples, num_balls) true numbers (for the last round)
        rounds_pred_list: list of (n_samples, num_balls) arrays, one per round
        prev_pred: (n_samples, num_balls) array of previous predictions (optional)
        num_balls: number of balls (default 5)
        round_labels: list of labels for each round (optional)
        prev_label: label for previous predictions
    """
    stds = []
    labels = []
    logger.info("[PLOT DIAG] plot_multi_round_true_std y_true (first 5): %s", y_true[:5])
    if prev_pred is not None:
        logger.info("[PLOT DIAG] plot_multi_round_true_std prev_pred (first 5): %s", prev_pred[:5])
    for idx, y_pred in enumerate(rounds_pred_list):
        logger.info(f"[PLOT DIAG] plot_multi_round_true_std round {idx+1} y_pred (first 5): %s", y_pred[:5])
    true_stds = np.std(y_true, axis=0)[:num_balls]
    stds.append(true_stds)
    labels.append('True')
    if prev_pred is not None:
        prev_stds = np.std(prev_pred, axis=0)[:num_balls]
        stds.append(prev_stds)
        labels.append(prev_label)
    # Ensure unique labels
    used_labels = set(labels)
    def make_unique(label):
        orig = label
        count = 2
        while label in used_labels:
            label = f"{orig} ({count})"
            count += 1
        used_labels.add(label)
        return label
    for idx, y_pred in enumerate(rounds_pred_list):
        round_stds = np.std(y_pred, axis=0)[:num_balls]
        stds.append(round_stds)
        if round_labels and idx < len(round_labels):
            label = round_labels[idx]
        else:
            label = f'Round {idx+1}'
        labels.append(make_unique(label))
    stds = np.array(stds)
    plt.figure(figsize=(10, 6))
    for i in range(stds.shape[1]):
        plt.plot(labels, stds[:, i], marker='o', label=f'Ball {i+1}')
    plt.title('True Std per Ball Across Rounds and Previous Runs')
    plt.xlabel('Source')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_multi_round_pred_std(y_true, rounds_pred_list, prev_pred=None, num_balls=5, round_labels=None, prev_label='Previous'):
    """
    Plot the predicted standard deviation for each ball across rounds and previous runs.
    """
    stds = []
    labels = []
    logger.info("[PLOT DIAG] plot_multi_round_pred_std y_true (first 5): %s", y_true[:5])
    if prev_pred is not None:
        logger.info("[PLOT DIAG] plot_multi_round_pred_std prev_pred (first 5): %s", prev_pred[:5])
    for idx, y_pred in enumerate(rounds_pred_list):
        logger.info(f"[PLOT DIAG] plot_multi_round_pred_std round {idx+1} y_pred (first 5): %s", y_pred[:5])
    true_stds = np.std(y_true, axis=0)[:num_balls]
    stds.append(true_stds)
    labels.append('True')
    if prev_pred is not None:
        prev_stds = np.std(prev_pred, axis=0)[:num_balls]
        stds.append(prev_stds)
        labels.append(prev_label)
    # Ensure unique labels
    used_labels = set(labels)
    def make_unique(label):
        orig = label
        count = 2
        while label in used_labels:
            label = f"{orig} ({count})"
            count += 1
        used_labels.add(label)
        return label
    for idx, y_pred in enumerate(rounds_pred_list):
        round_stds = np.std(y_pred, axis=0)[:num_balls]
        stds.append(round_stds)
        if round_labels and idx < len(round_labels):
            label = round_labels[idx]
        else:
            label = f'Round {idx+1}'
        labels.append(make_unique(label))
    stds = np.array(stds)
    plt.figure(figsize=(10, 6))
    for i in range(stds.shape[1]):
        plt.plot(labels, stds[:, i], marker='o', label=f'Ball {i+1}')
    plt.title('Predicted Std per Ball Across Rounds and Previous Runs')
    plt.xlabel('Source')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.show()

def kl_divergence(p, q, n_classes=None):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # If n_classes is provided, treat p and q as integer arrays and convert to distributions
    if n_classes is not None:
        p = np.bincount(p.astype(int)-1, minlength=n_classes) / len(p)
        q = np.bincount(q.astype(int)-1, minlength=n_classes) / len(q)
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return np.sum(p * np.log(p / q))

def plot_multi_round_kl_divergence(y_true, rounds_pred_list, prev_pred=None, num_balls=5, n_classes=69, round_labels=None, prev_label='Previous'):
    """
    Plot the KL divergence between true and predicted distributions for each ball across rounds and previous runs.
    """
    kls = []
    labels = []
    kls.append([0.0 for _ in range(num_balls)])
    labels.append('True')
    def get_dist_matrix(arr):
        # arr: (n_samples, num_balls)
        return np.stack([np.bincount(arr[:, i]-1, minlength=n_classes) / arr.shape[0] for i in range(num_balls)], axis=0)
    true_dists = get_dist_matrix(y_true)
    if prev_pred is not None:
        prev_dists = get_dist_matrix(prev_pred)
        prev_kls = np.sum(true_dists * np.log(np.clip(true_dists / np.clip(prev_dists, 1e-12, 1), 1e-12, 1)), axis=1)
        kls.append(prev_kls)
        labels.append(prev_label)
    logger.info("[PLOT DIAG] plot_multi_round_kl_divergence y_true (first 5): %s", y_true[:5])
    if prev_pred is not None:
        logger.info("[PLOT DIAG] plot_multi_round_kl_divergence prev_pred (first 5): %s", prev_pred[:5])
    # Ensure unique labels
    used_labels = set(labels)
    def make_unique(label):
        orig = label
        count = 2
        while label in used_labels:
            label = f"{orig} ({count})"
            count += 1
        used_labels.add(label)
        return label
    for idx, y_pred in enumerate(rounds_pred_list):
        logger.info(f"[PLOT DIAG] plot_multi_round_kl_divergence round {idx+1} y_pred (first 5): %s", y_pred[:5])
        pred_dists = get_dist_matrix(y_pred)
        kl = np.sum(true_dists * np.log(np.clip(true_dists / np.clip(pred_dists, 1e-12, 1), 1e-12, 1)), axis=1)
        kls.append(kl)
        if round_labels and idx < len(round_labels):
            label = round_labels[idx]
        else:
            label = f'Round {idx+1}'
        labels.append(make_unique(label))
    kls = np.array(kls)
    plt.figure(figsize=(10, 6))
    for i in range(kls.shape[1]):
        plt.plot(labels, kls[:, i], marker='o', label=f'Ball {i+1}')
    plt.title('KL Divergence per Ball Across Rounds')
    plt.xlabel('Round')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.tight_layout()
    plt.show()
