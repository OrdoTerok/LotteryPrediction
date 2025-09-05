import numpy as np
from core.optimal_assignment import optimal_assignment

def test_optimal_assignment_shape():
    prob_matrix = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.1, 0.0, 0.9]])
    result = optimal_assignment(prob_matrix)
    assert len(result) == prob_matrix.shape[0]
    assert set(result) <= set(range(prob_matrix.shape[1]))

def test_optimal_assignment_no_repeats():
    prob_matrix = np.eye(4)
    result = optimal_assignment(prob_matrix)
    assert len(set(result)) == 4
