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
            if missing_classes:
                X_dummy = np.repeat(X[:1], len(missing_classes), axis=0)
                y_dummy = np.array(list(missing_classes))
                X_aug = np.concatenate([X, X_dummy], axis=0)
                y_aug = np.concatenate([y_col, y_dummy], axis=0)
                model.fit(X_aug, y_aug)
            else:
                model.fit(X, y_col)
        y6 = y_sixth[:, 0]
        missing_classes6 = set(range(num_sixth_classes)) - set(np.unique(y6))
        if missing_classes6:
            X_dummy6 = np.repeat(X[:1], len(missing_classes6), axis=0)
            y_dummy6 = np.array(list(missing_classes6))
            X_aug6 = np.concatenate([X, X_dummy6], axis=0)
            y_aug6 = np.concatenate([y6, y_dummy6], axis=0)
            model_sixth.fit(X_aug6, y_aug6)
        else:
            model_sixth.fit(X, y6)

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
                print(f"  [ERROR][LGBM] Model {idx} ({type(model)}): Exception during predict_proba: {e}")
                raise
        try:
            first_five_pred = np.stack(preds, axis=1)
        except Exception as e:
            print("[ERROR][LGBM] Exception during np.stack for first_five_pred:", e)
            raise
        try:
            sixth_pred = model_sixth.predict_proba(X)[:, np.newaxis, :]
        except Exception as e:
            print("[ERROR][LGBM] Exception during model_sixth.predict_proba:", e)
            raise
        return first_five_pred, sixth_pred
