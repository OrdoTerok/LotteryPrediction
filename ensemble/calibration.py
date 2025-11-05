"""
Calibration Module
=================
This module provides calibration utilities for model probability outputs in the LotteryPrediction pipeline.
Includes temperature scaling, Platt scaling, and isotonic regression calibration classes.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

class TemperatureScaler:
    """
    Temperature scaling for calibrating model logits to probabilities.
    """
    def __init__(self):
        """Initialize with default temperature 1.0."""
        self.temperature = 1.0
    def fit(self, logits, labels):
        """
        Fit the temperature parameter using negative log-likelihood minimization.
        Args:
            logits: Model logits (pre-softmax outputs).
            labels: True class labels.
        """
        from scipy.optimize import minimize
        def nll(temp):
            probs = softmax(logits / temp, axis=-1)
            n = probs.shape[0]
            idx = np.arange(n)
            return -np.mean(np.log(probs[idx, labels] + 1e-12))
        res = minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)])
        self.temperature = float(res.x[0])
    def transform(self, logits):
        """
        Transform logits to calibrated probabilities using learned temperature.
        Args:
            logits: Model logits (pre-softmax outputs).
        Returns:
            Calibrated probabilities (softmax output).
        """
        return softmax(logits / self.temperature, axis=-1)

class PlattScaler:
    """
    Platt scaling using logistic regression for probability calibration.
    """
    def __init__(self):
        """Initialize PlattScaler with no fitted model."""
        self.lr = None
    def fit(self, logits, labels):
        """
        Fit logistic regression to logits and labels.
        Args:
            logits: Model logits or features.
            labels: True class labels.
        """
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(logits, labels)
    def transform(self, logits):
        """
        Transform logits to calibrated probabilities using fitted logistic regression.
        Args:
            logits: Model logits or features.
        Returns:
            Calibrated probabilities.
        """
        return self.lr.predict_proba(logits)

class IsotonicCalibrator:
    """
    Isotonic regression calibration for probability outputs.
    """
    def __init__(self):
        """Initialize IsotonicCalibrator with no fitted regressors."""
        self.ir = None
    def fit(self, probs, labels):
        """
        Fit isotonic regression for each class probability column.
        Args:
            probs: Predicted probabilities (n_samples, n_classes).
            labels: True class labels.
        """
        self.ir = []
        for i in range(probs.shape[1]):
            ir_i = IsotonicRegression(out_of_bounds='clip')
            ir_i.fit(probs[:, i], (labels == i).astype(int))
            self.ir.append(ir_i)
    def transform(self, probs):
        """
        Transform probabilities using fitted isotonic regressors.
        Args:
            probs: Predicted probabilities (n_samples, n_classes).
        Returns:
            Calibrated and renormalized probabilities.
        """
        calibrated_cols = [ir_i.transform(probs[:, i]) for i, ir_i in enumerate(self.ir)]
        calibrated = np.stack(calibrated_cols, axis=1)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated
