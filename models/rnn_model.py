import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

class RNNModel:
    @staticmethod
    def build_rnn_model(hp=None, input_shape=(10, 6),
                        units=64, num_layers=1, dropout=0.2,
                        use_bidirectional=False, optimizer='adam', learning_rate=1e-3):
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
        x = inputs
        for i in range(num_layers):
            rnn_layer = layers.SimpleRNN(units, return_sequences=(i < num_layers - 1))
            if use_bidirectional:
                x = layers.Bidirectional(rnn_layer)(x)
            else:
                x = rnn_layer(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
        # Output for first five balls (1-69)
        first_five_dense = layers.Dense(5 * 69)(x)
        first_five_softmax = layers.Reshape((5, 69))(first_five_dense)
        first_five_softmax = layers.Softmax(axis=-1, name='first_five')(first_five_softmax)
        # Output for sixth ball (1-26) -- ensure shape (None, 1, 26)
        sixth_dense = layers.Dense(26)(x)
        sixth_reshape = layers.Reshape((1, 26))(sixth_dense)
        sixth_softmax = layers.Softmax(axis=-1, name='sixth')(sixth_reshape)
        model = Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        # Optimizer
        if optimizer_choice == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_choice == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            opt = tf.keras.optimizers.Nadam(learning_rate)
        # Over-prediction penalty (same as LSTM)
        import config
        penalty_weight = getattr(config, 'OVERCOUNT_PENALTY_WEIGHT', 0.0)
        def overcount_penalty(y_true, y_pred):
            true_counts = tf.reduce_sum(y_true, axis=[0, 1])
            pred_counts = tf.reduce_sum(y_pred, axis=[0, 1])
            excess = tf.nn.relu(pred_counts - true_counts)
            return tf.reduce_sum(tf.square(excess))

        def first_five_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            return ce + penalty_weight * penalty

        def sixth_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            return ce + penalty_weight * penalty

        if penalty_weight > 0:
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
        # For KerasTuner batch size
        if hp is not None:
            try:
                batch_size = hp.Choice('batch_size', [16, 32, 64])
            except Exception:
                batch_size = 32
            model._tuner_batch_size = batch_size
        return model
