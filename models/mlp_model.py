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
    if y_pred.shape[1] < 2:
        return 0.0
    pred_idx = tf.argmax(y_pred, axis=-1)
    penalty = 0.0
    for i in range(y_pred.shape[1]):
        for j in range(i+1, y_pred.shape[1]):
            penalty += tf.reduce_mean(tf.cast(tf.equal(pred_idx[:, i], pred_idx[:, j]), tf.float32))
    return penalty / (y_pred.shape[1] * (y_pred.shape[1] - 1) / 2)

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

import tensorflow as tf
import numpy as np
from models.base_model import BaseModel
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

class MLPModel(BaseModel):
    def __init__(self, input_shape, num_first=5, num_first_classes=69, num_sixth_classes=26, hidden_units=64, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_first = num_first
        self.num_first_classes = num_first_classes
        self.num_sixth_classes = num_sixth_classes
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        self.model = self._build_mlp_model()

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
        cb = callbacks if callbacks is not None else [self.early_stopping_callback]
        return self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=cb, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, batch_size=32, **kwargs):
        return self.model.evaluate(X, y, batch_size=batch_size, **kwargs)


        def anti_copying_penalty(y_pred, meta_features=None):
            # meta_features: (batch, balls) or None
            # y_pred: (batch, balls, classes)
            if meta_features is None:
                return 0.0
            pred_idx = tf.argmax(y_pred, axis=-1)  # (batch, balls)
            meta_features = tf.cast(meta_features, tf.int64)
            penalty = tf.reduce_mean(tf.cast(tf.equal(pred_idx, meta_features), tf.float32))
            return penalty

        def first_five_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            entropy_pen = entropy_reg(y_pred)
            jac = jaccard_loss(y_true, y_pred)
            dup_pen = duplicate_penalty(y_pred)
            div_pen = diversity_penalty(y_pred)
            # Extract meta-features from y_true if present (assume last channel is meta, or pass externally)
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

