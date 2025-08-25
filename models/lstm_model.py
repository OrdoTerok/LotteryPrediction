
import tensorflow as tf
import numpy as np
import keras_tuner as kt

class LSTMModel:
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
        if force_simple:
            units = 64
            x = tf.keras.layers.SimpleRNN(units=units, activation='relu')(inputs)
        else:
            # Add KerasTuner hyperparameters for stacking, dropout, and bidirectional
            units = 128 if force_low_units else hp.Int('units', min_value=64, max_value=256, step=32)
            use_bidirectional = hp.Boolean('bidirectional', default=True)
            dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
            # First LSTM layer (return_sequences=True for stacking)
            lstm1 = tf.keras.layers.LSTM(units=units, activation='relu', return_sequences=True)
            if use_bidirectional:
                x = tf.keras.layers.Bidirectional(lstm1)(inputs)
            else:
                x = lstm1(inputs)
            # Dropout after first LSTM
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            # Second LSTM layer (return_sequences=False)
            lstm2 = tf.keras.layers.LSTM(units=units, activation='relu', return_sequences=False)
            if use_bidirectional:
                x = tf.keras.layers.Bidirectional(lstm2)(x)
            else:
                x = lstm2(x)
            # Dropout after second LSTM
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        # First 5 numbers output
        first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
        first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        # Sixth number output
        sixth_dense = tf.keras.layers.Dense(num_sixth_classes)(x)
        sixth_reshaped = tf.keras.layers.Reshape((1, num_sixth_classes))(sixth_dense)
        sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_reshaped)
        model = tf.keras.Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        # Define batch_size and optimizer for KerasTuner
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

        # Custom loss with over-prediction penalty
        import config
        penalty_weight = getattr(config, 'OVERCOUNT_PENALTY_WEIGHT', 0.0)
        def overcount_penalty(y_true, y_pred):
            # y_true, y_pred: (batch, 5, 69) or (batch, 1, 26)
            # Sum over batch and balls to get predicted/true count for each number
            true_counts = tf.reduce_sum(y_true, axis=[0, 1])  # (n_classes,)
            pred_counts = tf.reduce_sum(y_pred, axis=[0, 1])  # (n_classes,)
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
        model._tuner_batch_size = batch_size  # For use in main.py
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
        # First 5 numbers output
        first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
        first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        # Sixth number output
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
        print("\nStarting model training...")
        model.fit(X_train_reshaped, {'first_five': y_train[0], 'sixth': y_train[1]}, epochs=20, batch_size=32, verbose=1)
        print("Model training complete.")
        return model
