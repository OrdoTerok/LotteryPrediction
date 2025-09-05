def kl_to_uniform_probs(self, probs):
    """
    Compute KL divergence to uniform for predicted probabilities.
    Args:
        probs: np.ndarray, shape (n_samples, n_classes) or (n_samples, num_balls, n_classes)
    Returns:
       float: mean KL divergence to uniform
    """
    from util.metrics import kl_to_uniform
    import numpy as np
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
models.rnn_model
---------------
RNN-based neural network model for lottery prediction. Supports custom loss, KerasTuner integration, and cross-validation.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging

class RNNModel:
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
            self.logger.info(f"[RNN][CV] Fold {fold+1}/{cv}...")
            X_train, X_val = X[train_idx], X[val_idx]
            if isinstance(y, dict):
                y_train = {k: v[train_idx] for k, v in y.items()}
                y_val = {k: v[val_idx] for k, v in y.items()}
            elif isinstance(y, (list, tuple)):
                y_train = [v[train_idx] for v in y]
                y_val = [v[val_idx] for v in y]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            model = RNNModel(self.model.input_shape[1:])
            model.fit(X_train, y_train, **kwargs)
            eval_result = model.evaluate(X_val, y_val, **kwargs)
            results.append(eval_result)
            self.logger.info(f"[RNN][CV] Fold {fold+1} result: {eval_result}")
        return results
    @staticmethod
    def build_rnn_model(hp=None, input_shape=(10, 6),
                        units=64, num_layers=1, dropout=0.2,
                        use_bidirectional=False, optimizer='adam', learning_rate=1e-3,
                        use_custom_loss=False, temp_max=None):
        """
        Build a simple RNN model for lottery prediction, compatible with ensembling.
        Args:
            hp: Optional KerasTuner HyperParameters for tuning.
            input_shape: Shape of the input data.
            units: Number of RNN units.
            num_layers: Number of RNN layers.
            dropout: Dropout rate.
            use_bidirectional: Whether to use bidirectional RNNs.
            optimizer: Optimizer type.
            learning_rate: Learning rate for optimizer.
            use_custom_loss: Whether to use a custom loss function.
        Returns:
            Compiled Keras model.
        """
        if hp is not None:
            units = hp.Choice('rnn_units', [32, 64, 128])
            num_layers = hp.Choice('rnn_layers', [1, 2, 3])
            dropout = hp.Float('rnn_dropout', 0.0, 0.5, step=0.1)
            use_bidirectional = hp.Boolean('rnn_bidirectional')
            optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam'])
            learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4, 5e-5])
        else:
            optimizer_choice = optimizer

        inputs = layers.Input(shape=input_shape)
        x = layers.GaussianNoise(0.5)(inputs)
        for i in range(num_layers):
            rnn_layer = layers.SimpleRNN(units, return_sequences=(i < num_layers - 1))
            if use_bidirectional:
                x = layers.Bidirectional(rnn_layer)(x)
            else:
                x = rnn_layer(x)
            x = layers.Dropout(max(dropout, 0.5))(x)
        first_five_dense = layers.Dense(5 * 69)(x)
        first_five_reshaped = layers.Reshape((5, 69))(first_five_dense)
        if temp_max is not None and temp_max > 0.0:
            first_five_scaled = layers.Lambda(lambda z: z / temp_max)(first_five_reshaped)
            first_five_softmax = layers.Softmax(axis=-1, name='first_five')(first_five_scaled)
        else:
            first_five_softmax = layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        sixth_dense = layers.Dense(26)(x)
        sixth_reshape = layers.Reshape((1, 26))(sixth_dense)
        if temp_max is not None and temp_max > 0.0:
            sixth_scaled = layers.Lambda(lambda z: z / temp_max)(sixth_reshape)
            sixth_softmax = layers.Softmax(axis=-1, name='sixth')(sixth_scaled)
        else:
            sixth_softmax = layers.Softmax(axis=-1, name='sixth')(sixth_reshape)
        model = Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        from tensorflow.keras.callbacks import EarlyStopping
        model.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        if optimizer_choice == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_choice == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            opt = tf.keras.optimizers.Nadam(learning_rate)

        if use_custom_loss:
            import config.config as config
            from tensorflow.keras import backend as K
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
                intersection = K.sum(K.minimum(y_true, y_pred), axis=-1)
                union = K.sum(K.maximum(y_true, y_pred), axis=-1)
                return 1.0 - K.mean(intersection / (union + 1e-8))

            def duplicate_penalty(y_pred):
                # Avoid division by zero for single-ball predictions (e.g., sixth ball)
                if y_pred.shape[1] < 2:
                    return 0.0
                pred_idx = tf.argmax(y_pred, axis=-1)
                unique, _, count = tf.unique_with_counts(tf.reshape(pred_idx, [-1]))
                penalty = tf.reduce_sum(tf.cast(count > 1, tf.float32) * (tf.cast(count, tf.float32) - 1)) / tf.cast(tf.size(pred_idx), tf.float32)
                return penalty

            def diversity_penalty(y_pred):
                pred_idx = tf.argmax(y_pred, axis=-1)
                unique, _, count = tf.unique_with_counts(tf.reshape(pred_idx, [-1]))
                penalty = tf.reduce_sum(tf.cast(count > 1, tf.float32) * (tf.cast(count, tf.float32) - 1)) / tf.cast(tf.size(pred_idx), tf.float32)
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
                optimizer=opt,
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
                optimizer=opt,
                loss={
                    'first_five': 'categorical_crossentropy',
                    'sixth': 'categorical_crossentropy'
                },
                metrics={
                    'first_five': 'accuracy',
                    'sixth': 'accuracy'
                }
            )
        if hp is not None:
            try:
                batch_size = hp.Choice('batch_size', [16, 32, 64])
            except Exception:
                batch_size = 32
            model._tuner_batch_size = batch_size
        return model

    def __init__(self, input_shape, hp=None, use_custom_loss=False, units=64, num_layers=1, dropout=0.2, use_bidirectional=False, optimizer='adam', learning_rate=1e-3, label_smoothing=None, temp_max=None):
        self.logger = logging.getLogger(__name__)
        self.temp_max = temp_max
        self.logger.info(f"[RNN] Creating model with input_shape={input_shape}, hp={hp}, use_custom_loss={use_custom_loss}, units={units}, num_layers={num_layers}, dropout={dropout}, use_bidirectional={use_bidirectional}, optimizer={optimizer}, learning_rate={learning_rate}, label_smoothing={label_smoothing}, temp_max={temp_max}")
        self.model = self.build_rnn_model(hp, input_shape, units, num_layers, dropout, use_bidirectional, optimizer, learning_rate, use_custom_loss, temp_max=temp_max)
        self.logger.info("[RNN] Model created.")

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
        self.logger.info(f"[RNN] Starting fit: X shape={X.shape}, y shapes={get_y_shapes(y)}")
        # Apply label smoothing if set
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
        # Always suppress batch logs unless explicitly overridden
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 0
        history = self.model.fit(X, y, **kwargs)
        self.logger.info("[RNN] Finished fit.")
        return history

    def predict(self, X, **kwargs):
        self.logger.info(f"[RNN] Starting prediction: X shape={X.shape}")
        try:
            preds = self.model.predict(X, **kwargs)
            if isinstance(preds, (list, tuple)):
                pred_shapes = [p.shape for p in preds]
            else:
                pred_shapes = preds.shape
            self.logger.info(f"[RNN] Prediction complete: output shapes={pred_shapes}")
            return preds
        except Exception as e:
            self.logger.error(f"[RNN][ERROR] Exception during predict: {e}")
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
        self.logger.info(f"[RNN] Starting evaluation: X shape={X.shape}, y shapes={get_y_shapes(y)}")
        results = self.model.evaluate(X, y, **kwargs)
        probs = self.model.predict(X)
        kl_uniform = self.kl_to_uniform_probs(probs)
        self.logger.info(f"[RNN] Evaluation complete: results={results}, KL-to-uniform={kl_uniform:.4f}")
        return {"results": results, "kl_to_uniform": kl_uniform}
