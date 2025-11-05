"""
Optimal assignment utility for LotteryPrediction.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

def optimal_assignment(prob_matrix):
    """
    Given a probability matrix of shape (num_balls, num_classes),
    returns the optimal assignment of classes to balls (no repeats) maximizing total probability.
    Returns: list of assigned class indices (length num_balls)
    """
    cost_matrix = -prob_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind
