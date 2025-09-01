import matplotlib.pyplot as plt
import numpy as np

def plot_multi_run_ball_distributions(history, num_balls=5, n_classes=69, title_prefix='Ball', save_path=None):
    """
    Plot ball distributions for all runs in history.
    """
    palette = plt.get_cmap('tab10')
    for i in range(num_balls):
        plt.figure(figsize=(12, 5))
        x = np.arange(1, n_classes + 1)
        for run_idx, run in enumerate(history):
            preds = np.array(run.get('first_five_pred_numbers', []))
            if preds.shape[0] == 0:
                continue
            color = palette(run_idx % 10)
            round_counts = np.bincount(preds[:, i] - 1, minlength=n_classes)
            plt.bar(x + run_idx*0.1, round_counts, width=0.1, color=color, alpha=0.6, label=f'Run {run_idx+1}')
        plt.title(f'{title_prefix} {i+1} Distribution Across Runs')
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}_ball_{i+1}.png')
        plt.show()

def plot_multi_run_true_std(history, num_balls=5, save_path=None):
    """
    Plot true std per ball for all runs in history.
    """
    plt.figure(figsize=(10, 6))
    palette = plt.get_cmap('tab10')
    for run_idx, run in enumerate(history):
        preds = np.array(run.get('first_five_pred_numbers', []))
        if preds.shape[0] == 0:
            continue
        stds = [np.std(preds[:, i]) for i in range(num_balls)]
        plt.plot(range(1, num_balls+1), stds, marker='o', label=f'Run {run_idx+1}', color=palette(run_idx % 10))
    plt.title('Predicted Std per Ball Across Runs')
    plt.xlabel('Ball')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_std.png')
    plt.show()

def kl_divergence(p, q, n_classes=69):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.bincount(p.astype(int)-1, minlength=n_classes) / len(p)
    q = np.bincount(q.astype(int)-1, minlength=n_classes) / len(q)
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return np.sum(p * np.log(p / q))

def plot_multi_run_kl_divergence(history, num_balls=5, n_classes=69, save_path=None):
    """
    Plot KL divergence per ball for all runs in history (vs. first run as reference).
    """
    if not history or len(history) < 2:
        print("Need at least two runs for KL divergence plot.")
        return
    ref_preds = np.array(history[0].get('first_five_pred_numbers', []))
    if ref_preds.shape[0] == 0:
        print("No reference predictions in first run.")
        return
    plt.figure(figsize=(10, 6))
    palette = plt.get_cmap('tab10')
    for run_idx, run in enumerate(history[1:], 1):
        preds = np.array(run.get('first_five_pred_numbers', []))
        if preds.shape[0] == 0:
            continue
        kls = [kl_divergence(ref_preds[:, i], preds[:, i], n_classes=n_classes) for i in range(num_balls)]
        plt.plot(range(1, num_balls+1), kls, marker='o', label=f'Run {run_idx+1}', color=palette(run_idx % 10))
    plt.title('KL Divergence per Ball vs. First Run')
    plt.xlabel('Ball')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_kl.png')
    plt.show()
