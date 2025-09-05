import numpy as np
from visualization import plot_utils_std
import matplotlib
matplotlib.use('Agg')

def test_plot_multi_round_true_std_runs():
    y_true = np.array([[1,2,3,4,5],[2,3,4,5,6]])
    rounds_pred_list = [np.array([[1,2,3,4,5],[2,3,4,5,6]]) for _ in range(2)]
    plot_utils_std.plot_multi_round_true_std(y_true, rounds_pred_list)
