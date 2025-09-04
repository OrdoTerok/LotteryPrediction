# models/base_model.py

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def cross_validate(self, X, y, cv=5, **kwargs):
        """
        Optional: Cross-validation interface. Should be overridden by subclasses.
        """
        raise NotImplementedError("cross_validate not implemented for this model.")
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        pass
