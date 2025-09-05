import numpy as np
from visualization import plot_utils_closest
import matplotlib
matplotlib.use('Agg')

def test_plot_closest_value_distribution_runs():
    history = [
        {'closest_history': [
            {'match_count': 2, 'true_count': 2, 'within_10_percent': True},
            {'match_count': 3, 'true_count': 3, 'within_10_percent': False}
        ]}
    ]
    plot_utils_closest.plot_closest_value_distribution(history)
