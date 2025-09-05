import numpy as np
from core import metrics

def test_kl_divergence_basic():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert np.isclose(metrics.kl_divergence(p, q), 0)

def test_kl_to_uniform_basic():
    p = np.array([[1.0, 0.0]])
    assert metrics.kl_to_uniform(p) >= 0

def test_smooth_labels_shape():
    y = np.eye(3)
    smoothed = metrics.smooth_labels(y, 0.1)
    assert smoothed.shape == y.shape

def test_mix_uniform_shape():
    y = np.eye(3)
    mixed = metrics.mix_uniform(y, 0.5)
    assert mixed.shape == y.shape
