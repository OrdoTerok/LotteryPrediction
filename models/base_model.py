"""
models.base_model
---------------
Abstract base class for all model types. Defines the interface for fit, predict, evaluate, and cross_validate.
"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def cross_validate(self, X, y, cv=5, **kwargs):
        """
        Perform cross-validation on the model.
        Args:
            X: Input features.
            y: Target values.
            cv: Number of cross-validation folds.
            **kwargs: Additional arguments for cross-validation.
        Returns:
            Cross-validation results or raises NotImplementedError if not implemented.
        """
        raise NotImplementedError("cross_validate not implemented for this model.")
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Fit the model to the training data.
        Args:
            X: Input features.
            y: Target values.
            **kwargs: Additional arguments for fitting.
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions using the trained model.
        Args:
            X: Input features.
            **kwargs: Additional arguments for prediction.
        Returns:
            Model predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model on the provided data.
        Args:
            X: Input features.
            y: Target values.
            **kwargs: Additional arguments for evaluation.
        Returns:
            Evaluation metrics or results.
        """
        pass
