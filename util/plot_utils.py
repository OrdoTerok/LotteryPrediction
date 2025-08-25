import matplotlib.pyplot as plt
import seaborn as sns

def plot_ball_distributions(y_true, y_pred, num_balls=5, n_classes=69, title_prefix='Ball'):
    for i in range(num_balls):
        plt.figure(figsize=(10, 4))
        sns.histplot(y_true[:, i], color='blue', label='True', kde=False, bins=n_classes, stat='count', alpha=0.5)
        sns.histplot(y_pred[:, i], color='red', label='Predicted', kde=False, bins=n_classes, stat='count', alpha=0.5)
        plt.title(f'{title_prefix} {i+1} Distribution (1-{n_classes})')
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_powerball_distribution(y_true, y_pred, n_classes=26):
    plt.figure(figsize=(10, 4))
    sns.histplot(y_true[:, 0], color='blue', label='True', kde=False, bins=n_classes, stat='count', alpha=0.5)
    sns.histplot(y_pred[:, 0], color='red', label='Predicted', kde=False, bins=n_classes, stat='count', alpha=0.5)
    plt.title('Powerball (6th Ball) Distribution (1-26)')
    plt.xlabel('Number')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
