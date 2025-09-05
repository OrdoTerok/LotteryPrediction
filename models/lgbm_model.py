class LightGBMModel(BaseModel):

    def kl_to_uniform_probs(self, probs):
        """
        Compute KL divergence to uniform for predicted probabilities.
        Args:
            probs: np.ndarray, shape (n_samples, n_classes) or (n_samples, num_balls, n_classes)
        Returns:
            float: mean KL divergence to uniform
        """
        from util.metrics import kl_to_uniform
        if probs.ndim == 3:
            return float(np.mean([kl_to_uniform(probs[:, i, :]) for i in range(probs.shape[1])]))
        return float(kl_to_uniform(probs))
    @staticmethod
    def tune_with_kerastuner(tuner, *args, **kwargs):
        """
        Run KerasTuner search with console output suppressed.
        """
        from core.log_utils import suppress_console
        suppress_console()
        return tuner.search(*args, **kwargs)

"""
models.lgbm_model
-----------------
LightGBM-based model for lottery prediction. Supports multiclass classification for each ball and cross-validation.
"""
import lightgbm as lgb
import numpy as np
from models.base_model import BaseModel
import logging

class LightGBMModel(BaseModel):
    def cross_validate(self, X, y, cv=5, **kwargs):
        """
        Perform K-fold cross-validation.
        Args:
            X: Input features.
            y: Target values.
            cv: Number of cross-validation folds.
            **kwargs: Additional arguments for fitting/evaluation.
        Returns:
            List of evaluation results for each fold.
        """
        from sklearn.model_selection import KFold
        results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger = logging.getLogger(__name__)
            logger.info(f"[LGBM][CV] Fold {fold+1}/{cv}...")
            X_train, X_val = X[train_idx], X[val_idx]
            if isinstance(y, (list, tuple)):
                y_train = [v[train_idx] for v in y]
                y_val = [v[val_idx] for v in y]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            model = LightGBMModel(self.num_first, self.num_first_classes, self.num_sixth_classes, self.params)
            model.fit(X_train, y_train, **kwargs)
            eval_result = model.evaluate(X_val, y_val, **kwargs)
            results.append(eval_result)
            logger.info(f"[LGBM][CV] Fold {fold+1} result: {eval_result}")
        return results
    def __init__(self, num_first=5, num_first_classes=69, num_sixth_classes=26, params=None):
        """
        Initialize the LightGBMModel.
        Args:
            num_first: Number of first balls.
            num_first_classes: Number of classes for first balls.
            num_sixth_classes: Number of classes for sixth ball.
            params: LightGBM parameters dictionary.
        """
        logger = logging.getLogger(__name__)
        self.num_first = num_first
        self.num_first_classes = num_first_classes
        self.num_sixth_classes = num_sixth_classes
        self.params = params if params is not None else {
            'objective': 'multiclass',
            'num_class': num_first_classes,
            'metric': 'multi_logloss',
            'verbosity': -1
        }
        logger.info(f"[LGBM] Creating {num_first} first-ball models and 1 sixth-ball model with params: {self.params}")
        self.models_first = [lgb.LGBMClassifier(**self.params) for _ in range(self.num_first)]
        params6 = self.params.copy()
        params6['num_class'] = num_sixth_classes
        self.model_sixth = lgb.LGBMClassifier(**params6)

    def fit(self, X, y, **kwargs):
        """
        Fit the LightGBM models to the training data.
        Args:
            X: Input features.
            y: Tuple or list of (y_first, y_sixth) targets.
            **kwargs: Additional arguments for fitting.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"[LGBM] Starting fit: X shape={X.shape}, y shapes={[arr.shape for arr in y] if isinstance(y, (list, tuple)) else y.shape}")
        y_first, y_sixth = y
        if y_first.ndim == 3:
            y_first = np.argmax(y_first, axis=-1)
        if y_sixth.ndim == 3:
            y_sixth = np.argmax(y_sixth, axis=-1)
        num_first_classes = self.num_first_classes
        num_sixth_classes = self.num_sixth_classes
        for i, model in enumerate(self.models_first):
            logger.info(f"[LGBM] Fitting model for ball {i+1}/{self.num_first}")
            y_col = y_first[:, i]
            missing_classes = set(range(num_first_classes)) - set(np.unique(y_col))
            fit_kwargs = {}
            fit_kwargs.update(kwargs)
            if 'eval_set' in fit_kwargs:
                fit_kwargs['early_stopping_rounds'] = 10
                fit_kwargs['eval_metric'] = 'multi_logloss'
            if missing_classes:
                logger.warning(f"[LGBM] Ball {i+1}: Missing classes {missing_classes}, adding dummy samples.")
                X_dummy = np.repeat(X[:1], len(missing_classes), axis=0)
                y_dummy = np.array(list(missing_classes))
                X_aug = np.concatenate([X, X_dummy], axis=0)
                y_aug = np.concatenate([y_col, y_dummy], axis=0)
                model.fit(X_aug, y_aug, **fit_kwargs)
            else:
                model.fit(X, y_col, **fit_kwargs)
            logger.info(f"[LGBM] Finished fitting model for ball {i+1}/{self.num_first}")
        y6 = y_sixth[:, 0]
        missing_classes6 = set(range(num_sixth_classes)) - set(np.unique(y6))
        fit_kwargs6 = {}
        fit_kwargs6.update(kwargs)
        if 'eval_set' in fit_kwargs6:
            fit_kwargs6['early_stopping_rounds'] = 10
            fit_kwargs6['eval_metric'] = 'multi_logloss'
        if missing_classes6:
            logger.warning(f"[LGBM] Sixth ball: Missing classes {missing_classes6}, adding dummy samples.")
            X_dummy6 = np.repeat(X[:1], len(missing_classes6), axis=0)
            y_dummy6 = np.array(list(missing_classes6))
            X_aug6 = np.concatenate([X, X_dummy6], axis=0)
            y_aug6 = np.concatenate([y6, y_dummy6], axis=0)
            self.model_sixth.fit(X_aug6, y_aug6, **fit_kwargs6)
        else:
            self.model_sixth.fit(X, y6, **fit_kwargs6)
        logger.info("[LGBM] Finished fit for all balls.")

    def predict(self, X, feature_names=None, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info(f"[LGBM] Starting prediction: X shape={X.shape}")
        import pandas as pd
        try:
            # Automatically flatten 3D input to 2D if needed
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
            if feature_names is not None and not isinstance(X, pd.DataFrame):
                if len(feature_names) == X.shape[1]:
                    X = pd.DataFrame(X, columns=feature_names)
                else:
                    X = pd.DataFrame(X)
            preds = []
            for idx, model in enumerate(self.models_first):
                try:
                    pred = model.predict_proba(X)
                    preds.append(pred)
                except Exception as e:
                    logger.error(f"  [ERROR][LGBM] Model {idx} ({type(model)}): Exception during predict_proba: {e}")
                    return None
            try:
                first_five_pred = np.stack(preds, axis=1)
            except Exception as e:
                logger.error(f"[ERROR][LGBM] Exception during np.stack for first_five_pred: {e}")
                return None
            try:
                sixth_pred = self.model_sixth.predict_proba(X)[:, np.newaxis, :]
            except Exception as e:
                logger.error(f"[ERROR][LGBM] Exception during model_sixth.predict_proba: {e}")
                return None
            logger.info(f"[LGBM] Prediction complete: first_five_pred shape={first_five_pred.shape}, sixth_pred shape={sixth_pred.shape}")
            return first_five_pred, sixth_pred
        except Exception as e:
            logger.error(f"[LGBM][ERROR] Exception during predict: {e}")
            return None

    def evaluate(self, X, y, feature_names=None, **kwargs):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("  [LGBM] Starting evaluation...")
        # Returns log loss for each ball and sixth
        from sklearn.metrics import log_loss
        first_five_pred, sixth_pred = self.predict(X, feature_names=feature_names, **kwargs)
        y_first, y_sixth = y
        if y_first.ndim == 3:
            y_first = np.argmax(y_first, axis=-1)
        if y_sixth.ndim == 3:
            y_sixth = np.argmax(y_sixth, axis=-1)
        losses = []
        for i in range(self.num_first):
            y_true = y_first[:, i]
            y_pred = first_five_pred[:, i, :]
            losses.append(log_loss(y_true, y_pred, labels=np.arange(self.num_first_classes)))
        y_true6 = y_sixth[:, 0]
        y_pred6 = sixth_pred[:, 0, :]
        losses.append(log_loss(y_true6, y_pred6, labels=np.arange(self.num_sixth_classes)))
        # Stack all probs for KL-to-uniform
        probs = np.concatenate([first_five_pred, sixth_pred], axis=1)
        kl_uniform = self.kl_to_uniform_probs(probs)
        logger.info(f"[LGBM] Evaluation complete: losses={losses}, KL-to-uniform={kl_uniform:.4f}")
        return {"losses": losses, "kl_to_uniform": kl_uniform}

    @staticmethod
    def fit_models(models_first, model_sixth, X, y):
        """
        Fit LightGBM models for each ball, ensuring all classes are present for each model.
        Args:
            models_first: list of LGBMClassifier for first five balls
            model_sixth: LGBMClassifier for sixth ball
            X: np.ndarray, shape (n_samples, n_features)
            y: tuple of (y_first, y_sixth), each as np.ndarray (one-hot or class indices)
        """
        y_first, y_sixth = y
        if y_first.ndim == 3:
            y_first = np.argmax(y_first, axis=-1)
        if y_sixth.ndim == 3:
            y_sixth = np.argmax(y_sixth, axis=-1)
        num_first_classes = 69
        num_sixth_classes = 26
        for i, model in enumerate(models_first):
            y_col = y_first[:, i]
            missing_classes = set(range(num_first_classes)) - set(np.unique(y_col))
            fit_kwargs = {'early_stopping_rounds': 10, 'eval_metric': 'multi_logloss', 'verbose': False}
            if missing_classes:
                X_dummy = np.repeat(X[:1], len(missing_classes), axis=0)
                y_dummy = np.array(list(missing_classes))
                X_aug = np.concatenate([X, X_dummy], axis=0)
                y_aug = np.concatenate([y_col, y_dummy], axis=0)
                model.fit(X_aug, y_aug, **fit_kwargs)
            else:
                model.fit(X, y_col, **fit_kwargs)
        y6 = y_sixth[:, 0]
        missing_classes6 = set(range(num_sixth_classes)) - set(np.unique(y6))
        fit_kwargs6 = {'early_stopping_rounds': 10, 'eval_metric': 'multi_logloss', 'verbose': False}
        if missing_classes6:
            X_dummy6 = np.repeat(X[:1], len(missing_classes6), axis=0)
            y_dummy6 = np.array(list(missing_classes6))
            X_aug6 = np.concatenate([X, X_dummy6], axis=0)
            y_aug6 = np.concatenate([y6, y_dummy6], axis=0)
            model_sixth.fit(X_aug6, y_aug6, **fit_kwargs6)
        else:
            model_sixth.fit(X, y6, **fit_kwargs6)

    @staticmethod
    def predict_proba(models_first, model_sixth, X, feature_names=None):
        """
        Predict class probabilities for each ball.
        Args:
            models_first: list of LGBMClassifier for first five balls
            model_sixth: LGBMClassifier for sixth ball
            X: np.ndarray, shape (n_samples, n_features)
        Returns:
            Tuple: (first_five_pred, sixth_pred), both as np.ndarray of probabilities
        """
        import pandas as pd
        # Only use feature_names if length matches X.shape[1]
        if feature_names is not None and not isinstance(X, pd.DataFrame):
            if len(feature_names) == X.shape[1]:
                X = pd.DataFrame(X, columns=feature_names)
            else:
                # Ignore feature_names if shape does not match
                X = pd.DataFrame(X)
        preds = []
        for idx, model in enumerate(models_first):
            try:
                pred = model.predict_proba(X)
                preds.append(pred)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"  [ERROR][LGBM] Model {idx} ({type(model)}): Exception during predict_proba: {e}")
                raise
        try:
            first_five_pred = np.stack(preds, axis=1)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[ERROR][LGBM] Exception during np.stack for first_five_pred: {e}")
            raise
        try:
            sixth_pred = model_sixth.predict_proba(X)[:, np.newaxis, :]
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[ERROR][LGBM] Exception during model_sixth.predict_proba: {e}")
            raise
        return first_five_pred, sixth_pred

    @staticmethod
    def create_models(num_first, num_first_classes, num_sixth_classes, params=None):
        """
        Builds one LightGBM multiclass classifier for each lottery ball (first five and sixth).
        Args:
            num_first: int, number of first balls
            num_first_classes: int, number of classes for first balls
            num_sixth_classes: int, number of classes for sixth ball
            params: dict, LightGBM parameters
        Returns:
            List of LightGBM models for first five balls, one for sixth ball
        """
        if params is None:
            params = {
                'objective': 'multiclass',
                'num_class': num_first_classes,
                'metric': 'multi_logloss',
                'verbosity': -1
            }
        models_first = [lgb.LGBMClassifier(**params) for _ in range(num_first)]
        params6 = params.copy()
        params6['num_class'] = num_sixth_classes
        model_sixth = lgb.LGBMClassifier(**params6)
        return models_first, model_sixth