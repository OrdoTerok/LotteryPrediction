import lightgbm as lgb
import numpy as np

class LightGBMModel:
    @staticmethod
    def build_lgbm_models(num_first=5, num_first_classes=69, num_sixth_classes=26, params=None):
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

    @staticmethod
    def fit(models_first, model_sixth, X, y):
        """
        Fit LightGBM models for each ball.
        Args:
            models_first: list of LGBMClassifier for first five balls
            model_sixth: LGBMClassifier for sixth ball
            X: np.ndarray, shape (n_samples, n_features)
            y: tuple of (y_first, y_sixth), each as np.ndarray (one-hot or class indices)
        """
        y_first, y_sixth = y
        # Convert one-hot to class indices if needed
        if y_first.ndim == 3:
            y_first = np.argmax(y_first, axis=-1)
        if y_sixth.ndim == 3:
            y_sixth = np.argmax(y_sixth, axis=-1)
        for i, model in enumerate(models_first):
            model.fit(X, y_first[:, i])
        model_sixth.fit(X, y_sixth[:, 0])

    @staticmethod
    def predict_proba(models_first, model_sixth, X):
        """
        Predict class probabilities for each ball.
        Args:
            models_first: list of LGBMClassifier for first five balls
            model_sixth: LGBMClassifier for sixth ball
            X: np.ndarray, shape (n_samples, n_features)
        Returns:
            Tuple: (first_five_pred, sixth_pred), both as np.ndarray of probabilities
        """
        first_five_pred = np.stack([model.predict_proba(X) for model in models_first], axis=1)
        sixth_pred = model_sixth.predict_proba(X)[:, np.newaxis, :]
        return first_five_pred, sixth_pred
