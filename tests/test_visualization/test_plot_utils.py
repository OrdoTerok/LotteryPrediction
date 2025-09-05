import numpy as np
import pytest
from visualization import plot_utils
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

def test_plot_multi_round_powerball_distribution_runs():
    y_true = np.array([[1], [2], [3], [4], [5]])
    rounds_pred_list = [np.array([[1], [2], [3], [4], [5]]) for _ in range(2)]
    # Should not raise
    plot_utils.plot_multi_round_powerball_distribution(y_true, rounds_pred_list)

def test_plot_multi_round_ball_distributions_runs():
    y_true = np.array([[1,2,3,4,5]])
    rounds_pred_list = [np.array([[1,2,3,4,5]]) for _ in range(2)]
    plot_utils.plot_multi_round_ball_distributions(y_true, rounds_pred_list)
