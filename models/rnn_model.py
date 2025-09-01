import tensorflow as tf
from tensorflow.keras import layers, Model

class RNNModel:
    @staticmethod
    def build_rnn_model(hp=None, input_shape=(10, 6),
                        units=64, num_layers=1, dropout=0.2,
                        use_bidirectional=False, optimizer='adam', learning_rate=1e-3,
                        use_custom_loss=False):
        """
        Build a simple RNN model for lottery prediction, compatible with ensembling.
        If hp (KerasTuner HyperParameters) is provided, use it for hyperparameter search.
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
        first_five_softmax = layers.Reshape((5, 69))(first_five_dense)
        first_five_softmax = layers.Softmax(axis=-1, name='first_five')(first_five_softmax)
        sixth_dense = layers.Dense(26)(x)
        sixth_reshape = layers.Reshape((1, 26))(sixth_dense)
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
            import config
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

    def __init__(self, input_shape, hp=None, use_custom_loss=False, units=64, num_layers=1, dropout=0.2, use_bidirectional=False, optimizer='adam', learning_rate=1e-3):
        self.model = self.build_rnn_model(hp, input_shape, units, num_layers, dropout, use_bidirectional, optimizer, learning_rate, use_custom_loss)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        return self.model.evaluate(X, y, **kwargs)
