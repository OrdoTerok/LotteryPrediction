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
    true_stds = [np.std(y_true[:, i]) for i in range(num_balls)]
    stds.append(true_stds)
    labels.append('True')
    if prev_pred is not None:
        prev_stds = [np.std(prev_pred[:, i]) for i in range(num_balls)]
        stds.append(prev_stds)
        labels.append(prev_label)
    for idx, y_pred in enumerate(rounds_pred_list):
        round_stds = [np.std(y_pred[:, i]) for i in range(num_balls)]
        stds.append(round_stds)
        if round_labels and idx < len(round_labels):
            labels.append(round_labels[idx])
        else:
            labels.append(f'Round {idx+1}')
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
    true_stds = [np.std(y_true[:, i]) for i in range(num_balls)]
    stds.append(true_stds)
    labels.append('True')
    if prev_pred is not None:
        prev_stds = [np.std(prev_pred[:, i]) for i in range(num_balls)]
        stds.append(prev_stds)
        labels.append(prev_label)
    for idx, y_pred in enumerate(rounds_pred_list):
        round_stds = [np.std(y_pred[:, i]) for i in range(num_balls)]
        stds.append(round_stds)
        if round_labels and idx < len(round_labels):
            labels.append(round_labels[idx])
        else:
            labels.append(f'Round {idx+1}')
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

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
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
    def get_dist(arr, i):
        return np.bincount(arr[:, i]-1, minlength=n_classes) / arr.shape[0]
    true_dists = [get_dist(y_true, i) for i in range(num_balls)]
    if prev_pred is not None:
        prev_kls = [kl_divergence(true_dists[i], get_dist(prev_pred, i)) for i in range(num_balls)]
        kls.append(prev_kls)
        labels.append(prev_label)
    for idx, y_pred in enumerate(rounds_pred_list):
        round_kls = [kl_divergence(true_dists[i], get_dist(y_pred, i)) for i in range(num_balls)]
        kls.append(round_kls)
        if round_labels and idx < len(round_labels):
            labels.append(round_labels[idx])
        else:
            labels.append(f'Round {idx+1}')
    kls = np.array(kls)
    plt.figure(figsize=(10, 6))
    for i in range(kls.shape[1]):
        plt.plot(labels, kls[:, i], marker='o', label=f'Ball {i+1}')
    plt.title('KL Divergence (True vs Predicted) per Ball Across Rounds and Previous Runs')
    plt.xlabel('Source')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.tight_layout()
    plt.show()
