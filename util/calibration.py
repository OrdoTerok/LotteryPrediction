import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0
    def fit(self, logits, labels):
        from scipy.optimize import minimize
        def nll(temp):
            probs = softmax(logits / temp, axis=-1)
            n = probs.shape[0]
            idx = np.arange(n)
            return -np.mean(np.log(probs[idx, labels] + 1e-12))
        res = minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)])
        self.temperature = float(res.x[0])
    def transform(self, logits):
        return softmax(logits / self.temperature, axis=-1)

class PlattScaler:
    def __init__(self):
        self.lr = None
    def fit(self, logits, labels):
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(logits, labels)
    def transform(self, logits):
        return self.lr.predict_proba(logits)

class IsotonicCalibrator:
    def __init__(self):
        self.ir = None
    def fit(self, probs, labels):
        self.ir = []
        for i in range(probs.shape[1]):
            ir_i = IsotonicRegression(out_of_bounds='clip')
            ir_i.fit(probs[:, i], (labels == i).astype(int))
            self.ir.append(ir_i)
    def transform(self, probs):
        # Vectorized transformation using list comprehension and np.stack
        calibrated_cols = [ir_i.transform(probs[:, i]) for i, ir_i in enumerate(self.ir)]
        calibrated = np.stack(calibrated_cols, axis=1)
        # Renormalize
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated
