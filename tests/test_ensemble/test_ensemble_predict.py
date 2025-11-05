import numpy as np
import pytest
from ensemble.ensemble_predict import ensemble_predict

class DummyModel:
    def __init__(self, pf, ps):
        self._pf = pf
        self._ps = ps
    def predict(self, X, verbose=0):
        return self._pf, self._ps

class DummyConfig:
    ENSEMBLE_STRATEGY = 'average'

def test_ensemble_predict_average():
    pf = np.ones((2, 5, 3))
    ps = np.ones((2, 1, 3))
    models = [DummyModel(pf, ps), DummyModel(pf * 2, ps * 2)]
    X = None
    config = DummyConfig()
    first, sixth = ensemble_predict(models, X, config)
    assert np.allclose(first, np.mean([pf, pf*2], axis=0))
    assert np.allclose(sixth, np.mean([ps, ps*2], axis=0))

def test_ensemble_predict_shape_mismatch():
    pf1 = np.ones((2, 5, 3))
    pf2 = np.ones((3, 5, 3))
    ps = np.ones((2, 1, 3))
    models = [DummyModel(pf1, ps), DummyModel(pf2, ps)]
    config = DummyConfig()
    with pytest.raises(RuntimeError):
        ensemble_predict(models, None, config)
