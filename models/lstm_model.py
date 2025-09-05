
"""
models.lstm_model
-----------------
LSTM-based neural network model for lottery prediction. Supports custom loss, KerasTuner integration, and cross-validation.
"""
import tensorflow as tf
import numpy as np
import keras_tuner as kt
from tensorflow.keras import backend as K
import logging

class LSTMModel:
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
            # For multi-ball, average over balls
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
            self.logger.info(f"[LSTM][CV] Fold {fold+1}/{cv}...")
            X_train, X_val = X[train_idx], X[val_idx]
            if isinstance(y, dict):
                y_train = {k: v[train_idx] for k, v in y.items()}
                y_val = {k: v[val_idx] for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                y_train = [v[train_idx] for v in y]
                y_val = [v[val_idx] for v in y]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            model = LSTMModel(self.model.input_shape[1:])
            model.fit(X_train, y_train, **kwargs)
            eval_result = model.evaluate(X_val, y_val, **kwargs)
            results.append(eval_result)
            self.logger.info(f"[LSTM][CV] Fold {fold+1} result: {eval_result}")
        return results
    class LoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, logger):
            """
            Initialize the LoggingCallback.
            Args:
                logger: Logger instance for logging training events.
            """
            super().__init__()
            self.logger = logger
        def on_epoch_end(self, epoch, logs=None):
            """
            Log the end of an epoch and metrics.
            Args:
                epoch: Current epoch number.
                logs: Optional logs dictionary.
            """
            log_str = f"Epoch {epoch+1} end: " + ', '.join([f"{k}: {v:.4f}" for k, v in (logs or {}).items()])
            self.logger.info(log_str)

    def __init__(self, input_shape, hp=None, use_custom_loss=False, force_low_units=False, force_simple=False,
                 units=None, dropout=None, learning_rate=None, label_smoothing=None, temp_max=None):
        """
        Initialize the LSTMModel.
        Args:
            input_shape: Shape of the input data.
            hp: Hyperparameters for model tuning.
            use_custom_loss: Whether to use a custom loss function.
            force_low_units: Force use of low number of units.
            force_simple: Use a simple model architecture.
            units: Number of LSTM units.
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            label_smoothing: Label smoothing parameter.
            temp_max: Maximum temperature for softmax.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[LSTM] Creating model with input_shape={input_shape}, use_custom_loss={use_custom_loss}, "
                         f"force_low_units={force_low_units}, force_simple={force_simple}, units={units}, "
                         f"dropout={dropout}, learning_rate={learning_rate}, label_smoothing={label_smoothing}, temp_max={temp_max}")
        self.temp_max = temp_max
        self.model = self.build_lstm_model(
            hp or kt.HyperParameters(),
            input_shape,
            use_custom_loss,
            force_low_units,
            force_simple,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            label_smoothing=label_smoothing,
            temp_max=temp_max
        )
        self.logger.info("[LSTM] Model created.")

    def fit(self, X, y, **kwargs):
        def get_y_shapes(y):
            if isinstance(y, dict):
                return {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                return [v.shape if hasattr(v, 'shape') else type(v) for v in y]
            elif hasattr(y, 'shape'):
                return y.shape
            else:
                return type(y)
        self.logger.info(f"[LSTM] Starting fit: X shape={X.shape}, y shapes={get_y_shapes(y)}")
        # Apply label smoothing if set
        if hasattr(self, 'model') and hasattr(self, 'model').__self__:
            label_smoothing = getattr(self.model.__self__, 'label_smoothing', None)
        else:
            label_smoothing = getattr(self, 'label_smoothing', None)
        if label_smoothing is not None and label_smoothing > 0.0:
            from util.metrics import smooth_labels
            if isinstance(y, dict):
                y = {k: smooth_labels(v, label_smoothing) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                y = [smooth_labels(v, label_smoothing) for v in y]
            else:
                y = smooth_labels(y, label_smoothing)
        # Apply mix_uniform if set
        mix_uniform_prob = getattr(self, 'mix_uniform_prob', None)
        if mix_uniform_prob is not None and mix_uniform_prob > 0.0:
            from util.metrics import mix_uniform
            if isinstance(y, dict):
                y = {k: mix_uniform(v, mix_uniform_prob) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                y = [mix_uniform(v, mix_uniform_prob) for v in y]
            else:
                y = mix_uniform(y, mix_uniform_prob)
        callbacks = kwargs.get('callbacks', [])
        callbacks = list(callbacks) + [LSTMModel.LoggingCallback(self.logger)]
        kwargs['callbacks'] = callbacks
        # Always suppress batch logs unless explicitly overridden
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 0
        history = self.model.fit(X, y, **kwargs)
        self.logger.info("[LSTM] Finished fit.")
        return history

    def predict(self, X, **kwargs):
        self.logger.info(f"[LSTM] Starting prediction: X shape={X.shape}")
        try:
            preds = self.model.predict(X, **kwargs)
            if isinstance(preds, (list, tuple)):
                pred_shapes = [p.shape for p in preds]
            else:
                pred_shapes = preds.shape
            self.logger.info(f"[LSTM] Prediction complete: output shapes={pred_shapes}")
            return preds
        except Exception as e:
            self.logger.error(f"[LSTM][ERROR] Exception during predict: {e}")
            return None

    def evaluate(self, X, y, **kwargs):
        def get_y_shapes(y):
            if isinstance(y, dict):
                return {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                return [v.shape if hasattr(v, 'shape') else type(v) for v in y]
            elif hasattr(y, 'shape'):
                return y.shape
            else:
                return type(y)
        self.logger.info(f"[LSTM] Starting evaluation: X shape={X.shape}, y shapes={get_y_shapes(y)}")
        results = self.model.evaluate(X, y, **kwargs)
        self.logger.info(f"[LSTM] Evaluation complete: results={results}")
        return results

    @staticmethod
    def build_lstm_model(hp, input_shape, use_custom_loss=False, force_low_units=False, force_simple=False,
                        units=None, dropout=None, learning_rate=None, label_smoothing=None, temp_max=None):
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
            _units = units if units is not None else 64
            x = tf.keras.layers.SimpleRNN(units=_units, activation='relu')(x)
        else:
            _units = units if units is not None else (128 if force_low_units else hp.Int('units', min_value=64, max_value=256, step=32))
            use_bidirectional = hp.Boolean('bidirectional', default=True)
            _dropout = dropout if dropout is not None else hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1, default=0.5)
            lstm1 = tf.keras.layers.LSTM(units=_units, activation='relu', return_sequences=True)
            if use_bidirectional:
                x = tf.keras.layers.Bidirectional(lstm1)(x)
            else:
                x = lstm1(x)
            x = tf.keras.layers.Dropout(_dropout)(x)
            lstm2 = tf.keras.layers.LSTM(units=_units, activation='relu', return_sequences=False)
            if use_bidirectional:
                x = tf.keras.layers.Bidirectional(lstm2)(x)
            else:
                x = lstm2(x)
            x = tf.keras.layers.Dropout(_dropout)(x)

        def diversity_penalty(y_pred):
            # Penalize repeated predictions in the batch
            pred_idx = tf.argmax(y_pred, axis=-1)
            unique, _, count = tf.unique_with_counts(tf.reshape(pred_idx, [-1]))
            penalty = tf.reduce_sum(tf.cast(count > 1, tf.float32) * (tf.cast(count, tf.float32) - 1)) / tf.cast(tf.size(pred_idx), tf.float32)
            return penalty
        first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
        # Temperature scaling for softmax
        if temp_max is not None and temp_max > 0.0:
            first_five_scaled = tf.keras.layers.Lambda(lambda z: z / temp_max)(first_five_reshaped)
            first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_scaled)
        else:
            first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        sixth_dense = tf.keras.layers.Dense(num_sixth_classes)(x)
        sixth_reshaped = tf.keras.layers.Reshape((1, num_sixth_classes))(sixth_dense)
        if temp_max is not None and temp_max > 0.0:
            sixth_scaled = tf.keras.layers.Lambda(lambda z: z / temp_max)(sixth_reshaped)
            sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_scaled)
        else:
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
            import config.config as config
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
            _learning_rate = learning_rate if learning_rate is not None else 1e-3
            optimizer = tf.keras.optimizers.Adam(_learning_rate)
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
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
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

    def fit(self, X, y, **kwargs):
        def get_y_shapes(y):
            if isinstance(y, dict):
                return {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                return [v.shape if hasattr(v, 'shape') else type(v) for v in y]
            elif hasattr(y, 'shape'):
                return y.shape
            else:
                return type(y)
        self.logger.info(f"[LSTM] Starting fit: X shape={X.shape}, y shapes={get_y_shapes(y)}")
        callbacks = kwargs.get('callbacks', [])
        callbacks = list(callbacks) + [LSTMModel.LoggingCallback(self.logger)]
        kwargs['callbacks'] = callbacks
        history = self.model.fit(X, y, **kwargs)
        self.logger.info("[LSTM] Finished fit.")
        return history

    def predict(self, X, **kwargs):
        self.logger.info(f"[LSTM] Starting prediction: X shape={X.shape}")
        try:
            preds = self.model.predict(X, **kwargs)
            if isinstance(preds, (list, tuple)):
                pred_shapes = [p.shape for p in preds]
            else:
                pred_shapes = preds.shape
            self.logger.info(f"[LSTM] Prediction complete: output shapes={pred_shapes}")
            return preds
        except Exception as e:
            self.logger.error(f"[LSTM][ERROR] Exception during predict: {e}")
            return None

    def evaluate(self, X, y, **kwargs):
        # Remove training-only arguments from kwargs for evaluate
        for arg in ["epochs", "batch_size", "validation_split", "verbose"]:
            kwargs.pop(arg, None)
        def get_y_shapes(y):
            if isinstance(y, dict):
                return {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                return [v.shape if hasattr(v, 'shape') else type(v) for v in y]
            elif hasattr(y, 'shape'):
                return y.shape
            else:
                return type(y)
        self.logger.info(f"[LSTM] Starting evaluation: X shape={X.shape}, y shapes={get_y_shapes(y)}")
        results = self.model.evaluate(X, y, **kwargs)
        # Compute KL to uniform for predictions
        probs = self.model.predict(X)
        kl_uniform = self.kl_to_uniform_probs(probs)
        self.logger.info(f"[LSTM] Evaluation complete: results={results}, KL-to-uniform={kl_uniform:.4f}")
        return {"results": results, "kl_to_uniform": kl_uniform}
