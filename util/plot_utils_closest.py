import matplotlib.pyplot as plt
import numpy as np

def plot_closest_history_vs_true(history, save_path=None):
    """
    Plot the number of matches per sample in closest_history vs. true count across multiple runs.
    Each run is shown as a different color/marker.
    """
    plt.figure(figsize=(14, 7))
    colors = plt.cm.get_cmap('tab10', len(history))
    for run_idx, run in enumerate(history):
        closest_history = run.get('closest_history', [])
        if not closest_history:
            continue
        match_counts = [entry['match_count'] for entry in closest_history]
        true_counts = [entry['true_count'] for entry in closest_history]
        within_margin = [entry['within_10_percent'] for entry in closest_history]
        indices = np.arange(len(match_counts))
        # Plot true counts only for the first run (assume same for all)
        if run_idx == 0:
            plt.plot(indices, true_counts, label='True Count', color='black', linestyle='--', linewidth=2)
        plt.scatter(
            indices,
            match_counts,
            c=[colors(run_idx)]*len(match_counts),
            marker='o',
            label=f'Predicted Match Count (Run {run_idx+1})',
            alpha=0.6
        )
        # Optionally, highlight within-margin points
        for idx, (x, y, w) in enumerate(zip(indices, match_counts, within_margin)):
            if w:
                plt.scatter(x, y, c=[colors(run_idx)], marker='*', s=80, edgecolor='k', linewidths=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Count')
    plt.title('Closest Prediction Match Count vs. True Count (All Runs)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
