import tensorflow as tf
import numpy as np
import keras_tuner as kt
from tensorflow.keras import backend as K

class LSTMModel:
    class LoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, logger):
            super().__init__()
            self.logger = logger
        def on_epoch_begin(self, epoch, logs=None):
            self.logger.info(f"Epoch {epoch+1} started.")
        def on_epoch_end(self, epoch, logs=None):
            log_str = f"Epoch {epoch+1} end: " + ', '.join([f"{k}: {v:.4f}" for k, v in (logs or {}).items()])
            self.logger.info(log_str)
        def on_train_batch_end(self, batch, logs=None):
            log_str = f"Batch {batch+1}: " + ', '.join([f"{k}: {v:.4f}" for k, v in (logs or {}).items()])
            self.logger.info(log_str)
    @staticmethod
    def build_lstm_model(hp, input_shape, use_custom_loss=False, force_low_units=False, force_simple=False):
        """
        Builds and compiles an LSTM model for KerasTuner hypertuning, with two outputs:
        - first_five: (5, 69) softmax for the first 5 numbers
        - sixth: (1, 26) softmax for the Powerball
        Args:
            hp: HyperParameters object from keras_tuner
            input_shape: tuple, shape of the input (timesteps, features)
        Returns:
            Compiled Keras model
        """
        num_first = 5
        num_first_classes = 69
        num_sixth_classes = 26
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.GaussianNoise(0.5)(inputs)
        if force_simple:
            units = 64
            x = tf.keras.layers.SimpleRNN(units=units, activation='relu')(x)
        else:
            units = 128 if force_low_units else hp.Int('units', min_value=64, max_value=256, step=32)
            use_bidirectional = hp.Boolean('bidirectional', default=True)
            dropout_rate = hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1, default=0.5)
            lstm1 = tf.keras.layers.LSTM(units=units, activation='relu', return_sequences=True)
            if use_bidirectional:
                x = tf.keras.layers.Bidirectional(lstm1)(x)
            else:
                x = lstm1(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            lstm2 = tf.keras.layers.LSTM(units=units, activation='relu', return_sequences=False)
            if use_bidirectional:
                x = tf.keras.layers.Bidirectional(lstm2)(x)
            else:
                x = lstm2(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        def diversity_penalty(y_pred):
            # Penalize repeated predictions in the batch
            pred_idx = tf.argmax(y_pred, axis=-1)
            unique, _, count = tf.unique_with_counts(tf.reshape(pred_idx, [-1]))
            penalty = tf.reduce_sum(tf.cast(count > 1, tf.float32) * (tf.cast(count, tf.float32) - 1)) / tf.cast(tf.size(pred_idx), tf.float32)
            return penalty
        first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
        first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        sixth_dense = tf.keras.layers.Dense(num_sixth_classes)(x)
        sixth_reshaped = tf.keras.layers.Reshape((1, num_sixth_classes))(sixth_dense)
        sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_reshaped)
        model = tf.keras.Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        try:
            batch_size = hp.Choice('batch_size', [16, 32, 64])
        except Exception:
            batch_size = 32
        try:
            optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam'])
            learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4, 5e-5])
        except Exception:
            optimizer_choice = 'adam'
            learning_rate = 1e-3
        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            optimizer = tf.keras.optimizers.Nadam(learning_rate)

        if use_custom_loss:
            import config
            penalty_weight = getattr(config, 'OVERCOUNT_PENALTY_WEIGHT', 0.0)
            entropy_penalty_weight = getattr(config, 'ENTROPY_PENALTY_WEIGHT', 0.0)
            jaccard_weight = getattr(config, 'JACCARD_LOSS_WEIGHT', 0.0)
            duplicate_penalty_weight = getattr(config, 'DUPLICATE_PENALTY_WEIGHT', 0.0)

            def overcount_penalty(y_true, y_pred):
                true_counts = tf.reduce_sum(y_true, axis=1)
                pred_counts = tf.reduce_sum(y_pred, axis=1)
                excess = tf.nn.relu(pred_counts - true_counts)
                penalty = tf.reduce_mean(tf.reduce_sum(tf.square(excess), axis=-1))
                return penalty

            def entropy_reg(y_pred):
                ent = -K.sum(y_pred * K.log(y_pred + 1e-8), axis=-1)
                return K.mean(ent)

            def jaccard_loss(y_true, y_pred):
                # y_true, y_pred: (batch, balls, classes)
                y_true_bin = tf.cast(y_true > 0, tf.float32)
                y_pred_bin = tf.cast(y_pred == tf.reduce_max(y_pred, axis=-1, keepdims=True), tf.float32)
                intersection = tf.reduce_sum(y_true_bin * y_pred_bin, axis=[1,2])
                union = tf.reduce_sum(tf.cast((y_true_bin + y_pred_bin) > 0, tf.float32), axis=[1,2])
                jaccard = 1.0 - intersection / (union + 1e-8)
                return tf.reduce_mean(jaccard)

            def duplicate_penalty(y_pred):
                # Penalize if same class is predicted for multiple balls in a ticket
                # y_pred: (batch, balls, classes)
                if y_pred.shape[1] < 2:
                    return 0.0
                pred_idx = tf.argmax(y_pred, axis=-1)  # (batch, balls)
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

            model.compile(
                optimizer=optimizer,
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
                optimizer=optimizer,
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

    @staticmethod
    def build_and_train_lstm(X_train: np.ndarray, y_train: tuple):
        """
        Builds, compiles, and trains a simple LSTM model (non-hypertuned, legacy method) for two outputs.
        """
        num_first = 5
        num_first_classes = 69
        num_sixth_classes = 26
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1, num_first + 1)
        inputs = tf.keras.Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
        x = tf.keras.layers.LSTM(50, activation='relu')(inputs)
        first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
        first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        sixth_dense = tf.keras.layers.Dense(num_sixth_classes)(x)
        sixth_reshaped = tf.keras.layers.Reshape((1, num_sixth_classes))(sixth_dense)
        sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_reshaped)
        model = tf.keras.Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        model.compile(
            optimizer='adam',
            loss={
                'first_five': 'categorical_crossentropy',
                'sixth': 'categorical_crossentropy'
            },
            metrics={
                'first_five': 'accuracy',
                'sixth': 'accuracy'
            }
        )
        import logging
        logger = logging.getLogger(__name__)
        logger.info("\nStarting model training...")
        callback = LSTMModel.LoggingCallback(logger)
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        model.fit(
            X_train_reshaped,
            {'first_five': y_train[0], 'sixth': y_train[1]},
            epochs=20,
            batch_size=32,
            verbose=0,
            callbacks=[callback, early_stop]
        )
        logger.info("Model training complete.")
        return model

    def __init__(self, input_shape, hp=None, use_custom_loss=False, force_low_units=False, force_simple=False):
        self.model = self.build_lstm_model(hp or kt.HyperParameters(), input_shape, use_custom_loss, force_low_units, force_simple)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        return self.model.evaluate(X, y, **kwargs)
