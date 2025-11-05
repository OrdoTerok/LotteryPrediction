import numpy as np
from pipeline.iterative_stacking import run_iterative_stacking

def test_run_iterative_stacking_runs():
    class DummyConfig:
        ITERATIVE_STACKING = True
        ITERATIVE_STACKING_ROUNDS = 2
    train_df = test_df = np.zeros((10, 7))
    y_true_first_five = np.zeros((10, 5))
    y_true_sixth = np.zeros((10, 1))
    # Should not raise
    rounds_first_five, rounds_sixth, round_labels = run_iterative_stacking(
        train_df, test_df, DummyConfig(), y_true_first_five, y_true_sixth)
    assert isinstance(rounds_first_five, list)
    assert isinstance(rounds_sixth, list)
    assert isinstance(round_labels, list)
