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
        first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
        first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
        first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
        sixth_dense = tf.keras.layers.Dense(num_sixth_classes)(x)
        sixth_reshaped = tf.keras.layers.Reshape((1, num_sixth_classes))(sixth_dense)
        sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_reshaped)
        model = tf.keras.Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
        # Optionally use custom loss with overcount penalty
        import config
        use_custom_loss = getattr(config, 'MLP_USE_CUSTOM_LOSS', False)
        penalty_weight = getattr(config, 'OVERCOUNT_PENALTY_WEIGHT', 0.0)
        entropy_penalty_weight = getattr(config, 'ENTROPY_PENALTY_WEIGHT', 0.0)
        import tensorflow.keras.backend as K
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
            return -K.mean(ent)
        def jaccard_loss(y_true, y_pred):
            y_true_bin = tf.cast(y_true > 0, tf.float32)
            y_pred_bin = tf.cast(y_pred == tf.reduce_max(y_pred, axis=-1, keepdims=True), tf.float32)
            intersection = tf.reduce_sum(y_true_bin * y_pred_bin, axis=[1,2])
            union = tf.reduce_sum(tf.cast((y_true_bin + y_pred_bin) > 0, tf.float32), axis=[1,2])
            jaccard = 1.0 - intersection / (union + 1e-8)
            return tf.reduce_mean(jaccard)
        def duplicate_penalty(y_pred):
            pred_idx = tf.argmax(y_pred, axis=-1)
            penalty = 0.0
            for i in range(y_pred.shape[1]):
                for j in range(i+1, y_pred.shape[1]):
                    penalty += tf.reduce_mean(tf.cast(tf.equal(pred_idx[:, i], pred_idx[:, j]), tf.float32))
            return penalty / (y_pred.shape[1] * (y_pred.shape[1] - 1) / 2)
        def first_five_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            entropy_pen = entropy_reg(y_pred)
            jac = jaccard_loss(y_true, y_pred)
            dup_pen = duplicate_penalty(y_pred)
            return ce + penalty_weight * penalty + entropy_penalty_weight * entropy_pen + jaccard_weight * jac + duplicate_penalty_weight * dup_pen
        def sixth_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            ce = tf.reduce_mean(ce)
            penalty = overcount_penalty(y_true, y_pred)
            entropy_pen = entropy_reg(y_pred)
            jac = jaccard_loss(y_true, y_pred)
            dup_pen = duplicate_penalty(y_pred)
            return ce + penalty_weight * penalty + entropy_penalty_weight * entropy_pen + jaccard_weight * jac + duplicate_penalty_weight * dup_pen
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
