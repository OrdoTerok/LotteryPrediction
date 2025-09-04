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
        # Vectorized extraction using numpy
        arr = np.array([
            (entry['match_count'], entry['true_count'], entry['within_10_percent'])
            for entry in closest_history
        ])
        match_counts = arr[:, 0].astype(int)
        true_counts = arr[:, 1].astype(int)
        within_margin = arr[:, 2].astype(bool)
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
        # Optionally, highlight within-margin points (vectorized)
        highlight_idx = np.where(within_margin)[0]
        if highlight_idx.size > 0:
            plt.scatter(indices[highlight_idx], match_counts[highlight_idx], c=[colors(run_idx)], marker='*', s=80, edgecolor='k', linewidths=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Count')
    plt.title('Closest Prediction Match Count vs. True Count (All Runs)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
