"""
meta_optimization.outer_inner
-----------------------------
Hierarchical hyperparameter optimization: PSO/Bayesian (outer) + KerasTuner (inner).
This module is opt-in and does not affect the current workflow unless explicitly called.
"""
import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
# import your PSO/Bayesian optimizer here (example: pyswarms)
# import your data/model prep utilities as needed

def build_model(hp, input_shape):
    """
    Build a multi-output lottery prediction model compatible with KerasTuner.
    Outputs:
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
    
    # Hyperparameters to tune
    units = hp.Int('units', min_value=64, max_value=256, step=32)
    dropout = hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1, default=0.5)
    use_bidirectional = hp.Boolean('bidirectional', default=True)
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4, 5e-5])
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam'])
    
    # Build model
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.GaussianNoise(0.5)(inputs)
    
    # LSTM layers
    lstm1 = tf.keras.layers.LSTM(units=units, activation='relu', return_sequences=True)
    if use_bidirectional:
        x = tf.keras.layers.Bidirectional(lstm1)(x)
    else:
        x = lstm1(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    lstm2 = tf.keras.layers.LSTM(units=units, activation='relu', return_sequences=False)
    if use_bidirectional:
        x = tf.keras.layers.Bidirectional(lstm2)(x)
    else:
        x = lstm2(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # First five numbers output
    first_five_dense = tf.keras.layers.Dense(num_first * num_first_classes)(x)
    first_five_reshaped = tf.keras.layers.Reshape((num_first, num_first_classes))(first_five_dense)
    first_five_softmax = tf.keras.layers.Softmax(axis=-1, name='first_five')(first_five_reshaped)
    
    # Sixth number (Powerball) output
    sixth_dense = tf.keras.layers.Dense(num_sixth_classes)(x)
    sixth_reshaped = tf.keras.layers.Reshape((1, num_sixth_classes))(sixth_dense)
    sixth_softmax = tf.keras.layers.Softmax(axis=-1, name='sixth')(sixth_reshaped)
    
    # Create model with two outputs
    model = tf.keras.Model(inputs=inputs, outputs=[first_five_softmax, sixth_softmax])
    
    # Configure optimizer
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = tf.keras.optimizers.Nadam(learning_rate)
    
    # Compile with categorical crossentropy for each output
    model.compile(
        optimizer=optimizer,
        loss={
            'first_five': 'categorical_crossentropy',
            'sixth': 'categorical_crossentropy'
        },
        metrics={
            'first_five': ['accuracy'],
            'sixth': ['accuracy']
        }
    )
    return model

def tune_with_kerastuner(X_train, y_train, input_shape):
    """
    Run KerasTuner to find best hyperparameters for lottery model.
    Args:
        X_train: Training features
        y_train: Training targets (dict or list with 'first_five' and 'sixth' outputs)
        input_shape: Shape of input data
    Returns:
        Tuple of (best_model, best_hyperparameters)
    """
    # Use first_five accuracy as the primary objective (since it's the main prediction task)
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective=kt.Objective('val_first_five_accuracy', direction='max'),
        max_trials=5,
        directory='kt_dir',
        project_name='kt_inner'
    )
    tuner.search(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)
    return best_model, best_hp

def outer_objective(meta_params, data_prep_fn):
    """
    Objective function for outer optimization (PSO/Bayesian).
    Args:
        meta_params: [meta1, meta2, ...] (e.g., window size, feature selection, etc.)
        data_prep_fn: function that returns (X_train, y_train, input_shape, X_val, y_val) given meta_params
    Returns:
        Negative validation score (for minimization)
    """
    X_train, y_train, input_shape, X_val, y_val = data_prep_fn(meta_params)
    best_model, best_hp = tune_with_kerastuner(X_train, y_train, input_shape)
    
    # Evaluate returns a list for multi-output models: [total_loss, first_five_loss, sixth_loss, first_five_acc, sixth_acc]
    eval_results = best_model.evaluate(X_val, y_val, verbose=0)
    
    # Use first_five accuracy as the primary metric (index 3 in the results)
    if isinstance(eval_results, (list, tuple)) and len(eval_results) >= 4:
        val_score = eval_results[3]  # first_five_accuracy
    else:
        # Fallback: use total loss (index 0)
        val_score = -eval_results[0] if isinstance(eval_results, (list, tuple)) else -eval_results
    
    return -val_score  # minimize for PSO (so negate the accuracy)

def run_outer_inner_optimization(data_prep_fn, pso_class, bounds, n_particles=5, iters=3):
    """
    data_prep_fn: function(meta_params) -> (X_train, y_train, input_shape, X_val, y_val)
    pso_class: PSO optimizer class (e.g., pyswarms.single.GlobalBestPSO)
    bounds: (lower_bounds, upper_bounds) for meta-parameters
    """
    import logging
    from core.log_utils import SuppressOutput
    
    logger = logging.getLogger(__name__)
    
    def pso_eval(X):
        # X shape: (n_particles, n_meta_params)
        return np.array([outer_objective(p, data_prep_fn) for p in X])
    
    optimizer = pso_class(n_particles=n_particles, dimensions=len(bounds[0]), options={'c1':0.5, 'c2':0.3, 'w':0.9}, bounds=bounds)
    
    # Suppress PySwarms console output
    with SuppressOutput():
        best_cost, best_pos = optimizer.optimize(pso_eval, iters=iters)
    
    # Log to file instead of printing to console
    logger.info(f"Best meta-params: {best_pos}")
    
    return best_cost, best_pos

# Usage example (does not run unless called):
# from pyswarms.single import GlobalBestPSO
# def my_data_prep(meta_params):
#     ... # return (X_train, y_train, input_shape, X_val, y_val)
# bounds = (np.array([1, 0]), np.array([10, 1]))
# run_outer_inner_optimization(my_data_prep, GlobalBestPSO, bounds)
