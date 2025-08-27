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
    import matplotlib.pyplot as plt
    import numpy as np
    palette = plt.get_cmap('tab10')
    plt.figure(figsize=(12, 5))
    x = np.arange(1, n_classes + 1)
    width = 0.8 / (2 + len(rounds_pred_list))
    offsets = np.linspace(-width * (1 + len(rounds_pred_list)) / 2, width * (1 + len(rounds_pred_list)) / 2, 2 + len(rounds_pred_list))
    print("[PLOT DIAG] y_true (first 5):", y_true[:5])
    if prev_pred is not None:
        print("[PLOT DIAG] prev_pred (first 5):", prev_pred[:5])
    for idx, y_pred in enumerate(rounds_pred_list):
        print(f"[PLOT DIAG] round {idx+1} y_pred (first 5):", y_pred[:5])
    true_counts = np.bincount(y_true[:, 0] - 1, minlength=n_classes)
    plt.bar(x + offsets[0], true_counts, width=width, color='blue', label='True', align='center')
    idx_offset = 1
    if prev_pred is not None:
        prev_counts = np.bincount(prev_pred[:, 0] - 1, minlength=n_classes)
        plt.bar(x + offsets[1], prev_counts, width=width, color='black', label=prev_label, align='center')
        idx_offset += 1
    for idx, y_pred in enumerate(rounds_pred_list):
        label = round_labels[idx] if round_labels and idx < len(round_labels) else f'Round {idx+1}'
        color = palette((idx + 2) % 10)
        round_counts = np.bincount(y_pred[:, 0] - 1, minlength=n_classes)
        plt.bar(x + offsets[idx + idx_offset], round_counts, width=width, color=color, label=label, align='center')
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
    import matplotlib.pyplot as plt
    import numpy as np
    palette = plt.get_cmap('tab10')
    for i in range(num_balls):
        print(f"[PLOT DIAG] Ball {i+1} y_true (first 5):", y_true[:5, i])
        if prev_pred is not None:
            print(f"[PLOT DIAG] Ball {i+1} prev_pred (first 5):", prev_pred[:5, i])
        for idx, y_pred in enumerate(rounds_pred_list):
            print(f"[PLOT DIAG] Ball {i+1} round {idx+1} y_pred (first 5):", y_pred[:5, i])
        plt.figure(figsize=(12, 5))
        x = np.arange(1, n_classes + 1)
        width = 0.8 / (2 + len(rounds_pred_list))  # bar width
        offsets = np.linspace(-width * (1 + len(rounds_pred_list)) / 2, width * (1 + len(rounds_pred_list)) / 2, 2 + len(rounds_pred_list))
        true_counts = np.bincount(y_true[:, i] - 1, minlength=n_classes)
        plt.bar(x + offsets[0], true_counts, width=width, color='blue', label='True', align='center')
        idx_offset = 1
        if prev_pred is not None:
            prev_counts = np.bincount(prev_pred[:, i] - 1, minlength=n_classes)
            plt.bar(x + offsets[1], prev_counts, width=width, color='black', label=prev_label, align='center')
            idx_offset += 1
        for idx, y_pred in enumerate(rounds_pred_list):
            label = round_labels[idx] if round_labels and idx < len(round_labels) else f'Round {idx+1}'
            color = palette((idx + 2) % 10)
            round_counts = np.bincount(y_pred[:, i] - 1, minlength=n_classes)
            plt.bar(x + offsets[idx + idx_offset], round_counts, width=width, color=color, label=label, align='center')
        plt.title(f'{title_prefix} {i+1} Distribution (1-{n_classes})')
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()
import seaborn as sns


def plot_ball_distributions(y_true, y_pred, num_balls=5, n_classes=69, title_prefix='Ball'):
    import matplotlib.pyplot as plt
    for i in range(num_balls):
        ax = sns.histplot(y_true[:, i], color='blue', label='True', kde=False, bins=n_classes, stat='count', alpha=0.5)
        sns.histplot(y_pred[:, i], color='red', label='Predicted', kde=False, bins=n_classes, stat='count', alpha=0.5, ax=ax)
        ax.set(title=f'{title_prefix} {i+1} Distribution (1-{n_classes})', xlabel='Number', ylabel='Count')
        ax.legend()
        ax.figure.tight_layout()
        plt.show()

def plot_powerball_distribution(y_true, y_pred, n_classes=26):
    import matplotlib.pyplot as plt
    ax = sns.histplot(y_true[:, 0], color='blue', label='True', kde=False, bins=n_classes, stat='count', alpha=0.5)
    sns.histplot(y_pred[:, 0], color='red', label='Predicted', kde=False, bins=n_classes, stat='count', alpha=0.5, ax=ax)
    ax.set(title='Powerball (6th Ball) Distribution (1-26)', xlabel='Number', ylabel='Count')
    ax.legend()
    ax.figure.tight_layout()
    plt.show()
