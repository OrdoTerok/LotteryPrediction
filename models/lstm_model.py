
import tensorflow as tf
import numpy as np
import keras_tuner as kt

class LSTMModel:
    @staticmethod
    def build_lstm_model(hp, input_shape):
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
        x = tf.keras.layers.LSTM(
            units=hp.Int('units', min_value=16, max_value=128, step=16),
            activation='relu'
        )(inputs)
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
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            ),
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
