def plot_multi_round_true_pred_std(y_true, pred_rounds_list, true_rounds_list=None, prev_true=None, prev_pred=None, round_labels=None, prev_label='Previous'):
    """
    Plot true and predicted standard deviation for each ball (including Powerball) side by side for each round/source.
    Args:
        y_true: (n_samples, num_balls) true numbers (for the last round)
        pred_rounds_list: list of (n_samples, num_balls) arrays, one per round (predictions)
        true_rounds_list: list of (n_samples, num_balls) arrays, one per round (true values, optional)
        prev_true: (n_samples, num_balls) previous true values (optional)
        prev_pred: (n_samples, num_balls) previous predictions (optional)
        round_labels: list of labels for each round (optional)
        prev_label: label for previous predictions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    num_balls = y_true.shape[1]
    # Compute stds for true and predicted for each round/source
    stds_true = []
    stds_pred = []
    labels = []
    # True for current y_true
    stds_true.append(np.std(y_true, axis=0))
    stds_pred.append(np.full(num_balls, np.nan))  # No pred for 'True'
    labels.append('True')
    # Previous
    if prev_true is not None:
        stds_true.append(np.std(prev_true, axis=0))
        stds_pred.append(np.std(prev_pred, axis=0) if prev_pred is not None else np.full(num_balls, np.nan))
        labels.append(prev_label)
    # Rounds
    for idx, pred in enumerate(pred_rounds_list):
        stds_pred.append(np.std(pred, axis=0))
        if true_rounds_list is not None:
            stds_true.append(np.std(true_rounds_list[idx], axis=0))
        else:
            stds_true.append(np.full(num_balls, np.nan))
        if round_labels and idx < len(round_labels):
            labels.append(round_labels[idx])
        else:
            labels.append(f'Round {idx+1}')
    stds_true = np.array(stds_true)
    stds_pred = np.array(stds_pred)
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    for i in range(num_balls):
        offset = (i - num_balls/2) * width/num_balls
        plt.bar(x + offset, stds_true[:, i], width/(num_balls+1), label=f'True Ball {i+1}' if i < num_balls-1 else 'True Powerball', color=f'C{i}', alpha=0.6, hatch='/')
        plt.bar(x + offset + width/(2*num_balls), stds_pred[:, i], width/(num_balls+1), label=f'Pred Ball {i+1}' if i < num_balls-1 else 'Pred Powerball', color=f'C{i}', alpha=0.9)
    plt.xticks(x, labels, rotation=30)
    plt.title('True vs Predicted Std per Ball (All 6) Across Rounds and Previous Runs')
    plt.xlabel('Source')
    plt.ylabel('Standard Deviation')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()
"""
Standard deviation and KL-divergence plotting utilities for LotteryPrediction.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

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
    num_balls = y_true.shape[1]
    true_stds = np.std(y_true, axis=0)
    stds.append(true_stds)
    labels.append('True')
    if prev_pred is not None:
        prev_stds = np.std(prev_pred, axis=0)
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
        round_stds = np.std(y_pred, axis=0)
        stds.append(round_stds)
        if round_labels and idx < len(round_labels):
            label = round_labels[idx]
        else:
            label = f'Round {idx+1}'
        labels.append(make_unique(label))
    stds = np.array(stds)
    plt.figure(figsize=(10, 6))
    for i in range(stds.shape[1]):
        if i == stds.shape[1] - 1:
            plt.plot(labels, stds[:, i], marker='o', label='Powerball')
        else:
            plt.plot(labels, stds[:, i], marker='o', label=f'Ball {i+1}')
    plt.title('True Std per Ball (All 6) Across Rounds and Previous Runs')
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

def plot_multi_round_kl_divergence(y_true, rounds_pred_list, prev_pred=None, num_balls=5, n_classes=69, round_labels=None, prev_label='Previous'):
    """
    Plot the KL divergence between true and predicted distributions for each ball across rounds and previous runs.
    """
    kls = []
    labels = []
    kls.append([0.0 for _ in range(num_balls)])
    labels.append('True')
    # Support per-ball n_classes (list or int)
    if isinstance(n_classes, int):
        n_classes_list = [n_classes] * num_balls
    else:
        n_classes_list = n_classes
    def get_dist_matrix(arr):
        # arr: (n_samples, num_balls)
        dists = []
        max_len = max(n_classes_list)
        for i in range(arr.shape[1]):
            if i >= len(n_classes_list):
                logger.warning(f"[plot_utils_std] Skipping column {i}: n_classes_list missing entry (len={len(n_classes_list)})")
                continue
            dist = np.bincount(arr[:, i]-1, minlength=n_classes_list[i]) / arr.shape[0]
            # Pad dist to max_len with zeros if needed
            if dist.shape[0] < max_len:
                dist = np.pad(dist, (0, max_len - dist.shape[0]), mode='constant')
            dists.append(dist)
        # Now all dists have shape (max_len,)
        return np.stack(dists, axis=0)
    true_dists = get_dist_matrix(y_true)
    if true_dists is None:
        logger.warning("[plot_utils_std] KL plot skipped: true_dists could not be computed due to shape mismatch.")
        return
    if prev_pred is not None:
        prev_dists = get_dist_matrix(prev_pred)
        if prev_dists is None:
            logger.warning("[plot_utils_std] KL plot skipped: prev_dists could not be computed due to shape mismatch.")
            return
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
        if pred_dists is None:
            logger.warning(f"[plot_utils_std] KL plot skipped for round {idx+1}: pred_dists could not be computed due to shape mismatch.")
            continue
        # Compute KL for each ball, handling per-ball n_classes
        kl = np.zeros(num_balls)
        for i in range(num_balls):
            # Only compute KL if both distributions are valid (sum to 1)
            if np.all(true_dists[i] > 0) and np.all(pred_dists[i] > 0):
                kl[i] = np.sum(true_dists[i] * np.log(np.clip(true_dists[i] / np.clip(pred_dists[i], 1e-12, 1), 1e-12, 1)))
            else:
                kl[i] = np.nan
        kls.append(kl)
        if round_labels and idx < len(round_labels):
                label = str(round_labels[idx])  # Ensure label is a string
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
