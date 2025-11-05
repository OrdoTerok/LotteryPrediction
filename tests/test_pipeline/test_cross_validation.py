import pytest
from pipeline.cross_validation import cross_validate_model

class DummyModel:
    def cross_validate(self, X, y, cv=5, **kwargs):
        return [1 for _ in range(cv)]

def test_cross_validate_model_success():
    model = DummyModel()
    X, y = [0], [0]
    results = cross_validate_model(model, X, y, cv=3)
    assert results == [1, 1, 1]

def test_cross_validate_model_not_implemented():
    class NoCV:
        pass
    with pytest.raises(NotImplementedError):
        cross_validate_model(NoCV(), [0], [0])
