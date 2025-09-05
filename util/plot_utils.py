"""
DEPRECATED: Moved to visualization/plot_utils.py
"""
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
