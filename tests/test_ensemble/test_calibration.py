import numpy as np
from ensemble.calibration import TemperatureScaler, PlattScaler, IsotonicCalibrator

def test_temperature_scaler_fit_transform():
    logits = np.array([[2.0, 1.0], [1.0, 2.0]])
    labels = np.array([0, 1])
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    probs = scaler.transform(logits)
    assert probs.shape == logits.shape
    assert np.allclose(probs.sum(axis=1), 1)

def test_platt_scaler_fit_transform():
    logits = np.array([[2.0, 1.0], [1.0, 2.0]])
    labels = np.array([0, 1])
    scaler = PlattScaler()
    scaler.fit(logits, labels)
    probs = scaler.transform(logits)
    assert probs.shape == logits.shape
    assert np.allclose(probs.sum(axis=1), 1)

def test_isotonic_calibrator_fit_transform():
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    labels = np.array([0, 1, 0])
    calibrator = IsotonicCalibrator()
    calibrator.fit(probs, labels)
    calibrated = calibrator.transform(probs)
    assert calibrated.shape == probs.shape
    assert np.allclose(calibrated.sum(axis=1), 1)
