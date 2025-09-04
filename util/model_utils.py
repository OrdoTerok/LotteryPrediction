# Standard library imports
import os
import json

# Third-party imports
import numpy as np
import tensorflow as tf
try:
    from tensorflow import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
except ImportError:
    pass

# Local imports
import config.config as config

def get_results_history(cache=None):
    """
    Loads and caches the results_predictions_history.json file in memory or via cache utility.
    Returns the cached history on subsequent calls.
    """
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    if cache is not None:
        history = cache.get(history_path)
        if history is not None:
            return history
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    else:
        history = []
    if cache is not None:
        cache.set(history_path, history)
    return history

def ensemble_predict(models, X):
    """
    Ensemble predictions from multiple models using the strategy specified in config.ENSEMBLE_STRATEGY.
    Supported strategies:
      - 'average': simple mean of model predictions
      - 'weighted': weighted average (equal weights by default)
      - 'stacking': meta-learner (logistic regression) stacking
    """
    preds_first = []
    preds_sixth = []
    shapes_first = []
    shapes_sixth = []
    for idx, m in enumerate(models):
        try:
            pf, ps = m.predict(X, verbose=0)
            preds_first.append(pf)
            preds_sixth.append(ps)
            shapes_first.append(pf.shape)
            shapes_sixth.append(ps.shape)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"  Model {idx} ({type(m)}): Exception during prediction: {e}")
            preds_first.append(None)
            preds_sixth.append(None)
            shapes_first.append(None)
            shapes_sixth.append(None)
    valid_first = [s for s in shapes_first if s is not None]
    valid_sixth = [s for s in shapes_sixth if s is not None]
    import logging
    logger = logging.getLogger(__name__)
    if len(set(valid_first)) > 1 or len(set(valid_sixth)) > 1:
        logger.error(f"[ERROR][ensemble_predict] Prediction shape mismatch! first_five shapes: {shapes_first}, sixth shapes: {shapes_sixth}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch! See diagnostic output above.")

    strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'average').lower()
    if any(pf is None for pf in preds_first) or any(ps is None for ps in preds_sixth):
        logger.error("[ERROR][ensemble_predict] At least one prediction is None! Indices: %s %s", [i for i,pf in enumerate(preds_first) if pf is None], [i for i,ps in enumerate(preds_sixth) if ps is None])
        raise RuntimeError("[ensemble_predict] At least one prediction is None! See diagnostic output above.")
    valid_first_shapes = [pf.shape for pf in preds_first]
    valid_sixth_shapes = [ps.shape for ps in preds_sixth]
    if len(set(valid_first_shapes)) > 1 or len(set(valid_sixth_shapes)) > 1:
        logger.error(f"[ERROR][ensemble_predict] Prediction shape mismatch before aggregation! first_five shapes: {valid_first_shapes}, sixth shapes: {valid_sixth_shapes}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch before aggregation! See diagnostic output above.")
    if strategy == 'average':
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth
    elif strategy == 'weighted':
        weights = np.ones(len(models)) / len(models)
        weighted_first = np.tensordot(weights, np.array(preds_first), axes=1)
        weighted_sixth = np.tensordot(weights, np.array(preds_sixth), axes=1)
        return weighted_first, weighted_sixth
    elif strategy == 'stacking':
        from meta_learner import NNMetaLearner
        n_samples, n_balls, n_classes = preds_first[0].shape
        stacked_first = np.zeros((n_samples, n_balls, n_classes))
        # Hyperparameter grid for meta-learner
        hidden_units_grid = [16, 32]
        epochs_grid = [5, 10]
        for b in range(n_balls):
            X_stack = np.stack([pf[:, b, :] for pf in preds_first], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_first, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            best_acc = -1
            best_pred = None
            for hu in hidden_units_grid:
                for ep in epochs_grid:
                    meta = NNMetaLearner(input_dim=len(models), output_dim=n_classes, hidden_units=hu, epochs=ep)
                    try:
                        meta.fit(X_stack, y_stack)
                        preds = meta.predict_proba(X_stack)
                        acc = np.mean(np.argmax(preds, axis=1) == y_stack)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = preds
                    except Exception:
                        continue
            if best_pred is not None:
                stacked_first[:, b, :] = best_pred.reshape(n_samples, n_classes)
            else:
                stacked_first[:, b, :] = np.mean([pf[:, b, :] for pf in preds_first], axis=0)
        n_samples, n_balls, n_classes = preds_sixth[0].shape
        stacked_sixth = np.zeros((n_samples, n_balls, n_classes))
        for b in range(n_balls):
            X_stack = np.stack([ps[:, b, :] for ps in preds_sixth], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_sixth, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            best_acc = -1
            best_pred = None
            for hu in hidden_units_grid:
                for ep in epochs_grid:
                    meta = NNMetaLearner(input_dim=len(models), output_dim=n_classes, hidden_units=hu, epochs=ep)
                    try:
                        meta.fit(X_stack, y_stack)
                        preds = meta.predict_proba(X_stack)
                        acc = np.mean(np.argmax(preds, axis=1) == y_stack)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = preds
                    except Exception:
                        continue
            if best_pred is not None:
                stacked_sixth[:, b, :] = best_pred.reshape(n_samples, n_classes)
            else:
                stacked_sixth[:, b, :] = np.mean([ps[:, b, :] for ps in preds_sixth], axis=0)
        return stacked_first, stacked_sixth
    else:
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth
