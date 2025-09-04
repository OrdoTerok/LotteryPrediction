import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return entropy(p, q)

def kl_to_uniform(p):
    n = p.shape[-1]
    uniform = np.ones((1, n)) / n
    # Vectorized KL divergence for all rows in p
    p = np.asarray(p, dtype=np.float64)
    uniform = np.broadcast_to(uniform, p.shape)
    # Use axis=1 to compute KL for each row
    kls = np.sum(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1) / np.clip(uniform, 1e-12, 1)), axis=1)
    return np.mean(kls)

def smooth_labels(y, smoothing):
    y = np.asarray(y, dtype=np.float32)
    n_classes = y.shape[-1]
    return y * (1 - smoothing) + smoothing / n_classes

def mix_uniform(y, mix_prob):
    y = np.asarray(y, dtype=np.float32)
    n_classes = y.shape[-1]
    mask = np.random.rand(*y.shape[:-1]) < mix_prob
    uniform = np.ones_like(y) / n_classes
    y[mask] = uniform[mask]
    return y
