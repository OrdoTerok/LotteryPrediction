"""
Cross-Validation Utilities Module
================================
This module provides cross-validation utilities for the LotteryPrediction pipeline.
The main function, `cross_validate_model`, performs cross-validation using a model's `cross_validate` method if available.
"""
def cross_validate_model(model, X, y, cv=5, **kwargs):
    """
    Perform cross-validation using a model's cross_validate method if available.

    Args:
        model: Model object with a cross_validate method.
        X: Features.
        y: Labels.
        cv (int): Number of cross-validation folds.
        **kwargs: Additional keyword arguments for cross_validate.

    Returns:
        list: Per-fold evaluation results.

    Raises:
        NotImplementedError: If the model does not support cross-validation.
    """
    if hasattr(model, 'cross_validate'):
        return model.cross_validate(X, y, cv=cv, **kwargs)
    else:
        raise NotImplementedError(f"Model {type(model)} does not support cross-validation.")
