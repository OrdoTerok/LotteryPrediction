import tensorflow as tf
import numpy as np

class MLPModel:
    @staticmethod
    def build_mlp_model(input_shape, num_first=5, num_first_classes=69, num_sixth_classes=26, hidden_units=128, dropout_rate=0.2):
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
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
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
