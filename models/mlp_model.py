import logging
import tensorflow as tf
import numpy as np
from models.base_model import BaseModel
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

def overcount_penalty(y_true, y_pred):
    true_counts = tf.reduce_sum(y_true, axis=1)
    pred_counts = tf.reduce_sum(y_pred, axis=1)
    excess = tf.nn.relu(pred_counts - true_counts)
    return tf.reduce_mean(excess)

def entropy_reg(y_pred):
    ent = -K.sum(y_pred * K.log(y_pred + 1e-8), axis=-1)
    return K.mean(ent)

def jaccard_loss(y_true, y_pred):
    y_true_bin = tf.cast(y_true > 0, tf.float32)
    y_pred_bin = tf.cast(y_pred == tf.reduce_max(y_pred, axis=-1, keepdims=True), tf.float32)
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin, axis=[1,2])
    union = tf.reduce_sum(tf.cast((y_true_bin + y_pred_bin) > 0, tf.float32), axis=[1,2])
    jaccard = 1.0 - intersection / (union + 1e-8)
    return tf.reduce_mean(jaccard)

def duplicate_penalty(y_pred):
    # Vectorized duplicate penalty: count duplicates in each row
    pred_idx = tf.argmax(y_pred, axis=-1)  # (batch, balls)
    num_balls = tf.shape(pred_idx)[1]
    max_ball = 69  # Powerball first five max value
    # For each row, count how many times each value appears
    counts = tf.map_fn(
        lambda row: tf.math.bincount(row, minlength=max_ball, maxlength=max_ball, dtype=tf.int32),
        pred_idx,
        fn_output_signature=tf.TensorSpec(shape=(max_ball,), dtype=tf.int32)
    )
    # For each row, sum the number of duplicates (count > 1)
    duplicates = tf.reduce_sum(tf.nn.relu(tf.cast(counts > 1, tf.float32) * (tf.cast(counts, tf.float32) - 1)), axis=1)
    n = tf.cast(num_balls, tf.float32)
    denom = n * (n - 1) / 2.0
    penalty = tf.reduce_mean(duplicates / denom)
    return penalty

def anti_copying_penalty(y_pred, meta_features=None):
    if meta_features is None:
        return 0.0
    pred_idx = tf.argmax(y_pred, axis=-1)
    meta_features = tf.cast(meta_features, tf.int64)
    penalty = tf.reduce_mean(tf.cast(tf.equal(pred_idx, meta_features), tf.float32))
    return penalty

def diversity_penalty(y_pred):
    pred_idx = tf.argmax(y_pred, axis=-1)
    unique, _, count = tf.unique_with_counts(tf.reshape(pred_idx, [-1]))
    penalty = tf.reduce_sum(tf.cast(count > 1, tf.float32) * (tf.cast(count, tf.float32) - 1)) / tf.cast(tf.size(pred_idx), tf.float32)
    return penalty

class MLPModel(BaseModel):
    def cross_validate(self, X, y, cv=5, **kwargs):
        """
        Perform K-fold cross-validation. Returns list of per-fold evaluation results.
        """
        from sklearn.model_selection import KFold
        results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            self.logger.info(f"[MLP][CV] Fold {fold+1}/{cv}...")
            X_train, X_val = X[train_idx], X[val_idx]
            if isinstance(y, dict):
                y_train = {k: v[train_idx] for k, v in y.items()}
                y_val = {k: v[val_idx] for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                y_train = [v[train_idx] for v in y]
                y_val = [v[val_idx] for v in y]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            model = MLPModel(self.input_shape)
            model.fit(X_train, y_train, **kwargs)
            eval_result = model.evaluate(X_val, y_val, **kwargs)
            results.append(eval_result)
            self.logger.info(f"[MLP][CV] Fold {fold+1} result: {eval_result}")
        return results
    def __init__(self, input_shape, num_first=5, num_first_classes=69, num_sixth_classes=26, hidden_units=64, dropout_rate=0.5, learning_rate=None, label_smoothing=None, temp_max=None):
        self.logger = logging.getLogger(__name__)
        self.input_shape = input_shape
        self.num_first = num_first
        self.num_first_classes = num_first_classes
        self.num_sixth_classes = num_sixth_classes
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.logger.info(f"[MLP] Creating model with input_shape={input_shape}, num_first={num_first}, num_first_classes={num_first_classes}, num_sixth_classes={num_sixth_classes}, hidden_units={hidden_units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, label_smoothing={label_smoothing}, temp_max={temp_max}")
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        self.model = self._build_mlp_model()
        self.logger.info("[MLP] Model created.")

    def _build_mlp_model(self):
        """
        Builds and compiles a simple dense MLP model for lottery prediction.
        Args:
            input_shape: tuple, shape of the input (timesteps, features)
            num_first: int, number of first balls (default 5)
            num_first_classes: int, number of classes for first balls (default 69)
            num_sixth_classes: int, number of classes for sixth ball (default 26)
            hidden_units: int, number of units in hidden layers
            dropout_rate: float, dropout rate
        Returns:
            Compiled Keras model
        """
        # Define model input and layers
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.GaussianNoise(0.5)(inputs)
        x = tf.keras.layers.Flatten()(x)
        first_five_dense = tf.keras.layers.Dense(self.num_first * self.num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((self.num_first, self.num_first_classes))(first_five_dense)
        first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        sixth_dense = tf.keras.layers.Dense(self.num_sixth_classes)(x)
        sixth_reshaped = tf.keras.layers.Reshape((1, self.num_sixth_classes))(sixth_dense)
        sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_reshaped)
        model = tf.keras.Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        import config
        use_custom_loss = getattr(config, 'MLP_USE_CUSTOM_LOSS', False)
        penalty_weight = getattr(config, 'OVERCOUNT_PENALTY_WEIGHT', 0.0)
        entropy_penalty_weight = getattr(config, 'ENTROPY_PENALTY_WEIGHT', 0.0)
        jaccard_weight = getattr(config, 'JACCARD_LOSS_WEIGHT', 0.0)
        duplicate_penalty_weight = getattr(config, 'DUPLICATE_PENALTY_WEIGHT', 0.0)

        def first_five_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            entropy_pen = entropy_reg(y_pred)
            jac = jaccard_loss(y_true, y_pred)
            dup_pen = duplicate_penalty(y_pred)
            div_pen = diversity_penalty(y_pred)
            meta_features = None
            if hasattr(y_true, '_keras_mask') and hasattr(y_true._keras_mask, 'meta_features'):
                meta_features = y_true._keras_mask.meta_features
            anti_copy_pen = anti_copying_penalty(y_pred, meta_features)
            anti_copy_weight = getattr(config, 'ANTI_COPY_PENALTY_WEIGHT', 1.0)
            return ce + penalty_weight * penalty + entropy_penalty_weight * entropy_pen + jaccard_weight * jac + duplicate_penalty_weight * dup_pen + 2.0 * div_pen + anti_copy_weight * anti_copy_pen

        def sixth_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            entropy_pen = entropy_reg(y_pred)
            jac = jaccard_loss(y_true, y_pred)
            dup_pen = duplicate_penalty(y_pred)
            div_pen = diversity_penalty(y_pred)
            meta_features = None
            if hasattr(y_true, '_keras_mask') and hasattr(y_true._keras_mask, 'meta_features'):
                meta_features = y_true._keras_mask.meta_features
            anti_copy_pen = anti_copying_penalty(y_pred, meta_features)
            anti_copy_weight = getattr(config, 'ANTI_COPY_PENALTY_WEIGHT', 1.0)
            return ce + penalty_weight * penalty + entropy_penalty_weight * entropy_pen + jaccard_weight * jac + duplicate_penalty_weight * dup_pen + 2.0 * div_pen + anti_copy_weight * anti_copy_pen

        if use_custom_loss:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss={
                    'first_five': first_five_loss,
                    'sixth': sixth_loss
                },
                metrics={
                    'first_five': 'accuracy',
                    'sixth': 'accuracy'
                }
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss={
                    'first_five': 'categorical_crossentropy',
                    'sixth': 'categorical_crossentropy'
                },
                metrics={
                    'first_five': 'accuracy',
                    'sixth': 'accuracy'
                }
            )
        return model

    def fit(self, X, y, validation_data=None, epochs=100, batch_size=32, callbacks=None, **kwargs):
        def get_y_shapes(y):
            if isinstance(y, dict):
                return {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                return [v.shape if hasattr(v, 'shape') else type(v) for v in y]
            elif hasattr(y, 'shape'):
                return y.shape
            else:
                return type(y)
        def get_val_shapes(val):
            if val is None:
                return None
            if isinstance(val, tuple) and len(val) == 2:
                x_val, y_val = val
                return (x_val.shape if hasattr(x_val, 'shape') else type(x_val), get_y_shapes(y_val))
            return type(val)
        self.logger.info(f"[MLP] Starting fit: X shape={X.shape}, y shapes={get_y_shapes(y)}, validation_data shapes={get_val_shapes(validation_data)}")
        cb = callbacks if callbacks is not None else [self.early_stopping_callback]
        history = self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=cb, **kwargs)
        self.logger.info("[MLP] Finished fit.")
        return history

    def predict(self, X, **kwargs):
        self.logger.info(f"[MLP] Starting prediction: X shape={X.shape}")
        # Automatically flatten 3D input to 2D if needed
        if hasattr(X, 'ndim') and X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        preds = self.model.predict(X, **kwargs)
        if isinstance(preds, (list, tuple)):
            pred_shapes = [p.shape for p in preds]
        else:
            pred_shapes = preds.shape
        self.logger.info(f"[MLP] Prediction complete: output shapes={pred_shapes}")
        return preds

    def evaluate(self, X, y, batch_size=32, **kwargs):
        self.logger.info(f"[MLP] Starting evaluation: X shape={X.shape}, y shapes={[arr.shape for arr in y] if isinstance(y, (list, tuple)) else y.shape}")
        results = self.model.evaluate(X, y, batch_size=batch_size, **kwargs)
        self.logger.info(f"[MLP] Evaluation complete: results={results}")
        return results

