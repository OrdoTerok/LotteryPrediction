import numpy as np
from visualization import plot_utils_multi
import matplotlib
matplotlib.use('Agg')

def test_plot_multi_run_ball_distributions_runs():
    history = [
        {'first_five_pred_numbers': np.array([[1,2,3,4,5],[2,3,4,5,6]])},
        {'first_five_pred_numbers': np.array([[1,2,3,4,5],[2,3,4,5,6]])}
    ]
    plot_utils_multi.plot_multi_run_ball_distributions(history)

def test_plot_multi_run_true_std_runs():
    history = [
        {'first_five_pred_numbers': np.array([[1,2,3,4,5],[2,3,4,5,6]])}
    ]
    plot_utils_multi.plot_multi_run_true_std(history)
