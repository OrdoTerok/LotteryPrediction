"""
meta_optimization.outer_inner
-----------------------------
Hierarchical hyperparameter optimization: PSO/Bayesian (outer) + KerasTuner (inner).
This module is opt-in and does not affect the current workflow unless explicitly called.
"""
import numpy as np
import keras_tuner as kt
from tensorflow import keras
# import your PSO/Bayesian optimizer here (example: pyswarms)
# import your data/model prep utilities as needed

def build_model(hp, input_shape):
    # Example: tune units and dropout
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
    model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def tune_with_kerastuner(X_train, y_train, input_shape):
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
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
    # meta_params: [meta1, meta2, ...] (e.g., window size, feature selection, etc.)
    # data_prep_fn: function that returns (X_train, y_train, input_shape, X_val, y_val) given meta_params
    X_train, y_train, input_shape, X_val, y_val = data_prep_fn(meta_params)
    best_model, best_hp = tune_with_kerastuner(X_train, y_train, input_shape)
    val_score = best_model.evaluate(X_val, y_val, verbose=0)[1]  # e.g., accuracy
    return -val_score  # minimize for PSO

def run_outer_inner_optimization(data_prep_fn, pso_class, bounds, n_particles=5, iters=3):
    """
    data_prep_fn: function(meta_params) -> (X_train, y_train, input_shape, X_val, y_val)
    pso_class: PSO optimizer class (e.g., pyswarms.single.GlobalBestPSO)
    bounds: (lower_bounds, upper_bounds) for meta-parameters
    """
    def pso_eval(X):
        # X shape: (n_particles, n_meta_params)
        return np.array([outer_objective(p, data_prep_fn) for p in X])
    optimizer = pso_class(n_particles=n_particles, dimensions=len(bounds[0]), options={'c1':0.5, 'c2':0.3, 'w':0.9}, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(pso_eval, iters=iters)
    print("Best meta-params:", best_pos)
    return best_cost, best_pos

# Usage example (does not run unless called):
# from pyswarms.single import GlobalBestPSO
# def my_data_prep(meta_params):
#     ... # return (X_train, y_train, input_shape, X_val, y_val)
# bounds = (np.array([1, 0]), np.array([10, 1]))
# run_outer_inner_optimization(my_data_prep, GlobalBestPSO, bounds)
