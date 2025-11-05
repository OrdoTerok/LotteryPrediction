
# LotteryPrediction Project

## Overview
LotteryPrediction is a modular, extensible pipeline for predicting lottery numbers using deep learning (LSTM/RNN/MLP), LightGBM, meta-optimization (PSO/Bayesian), cross-validation, calibration, and flexible ensembling. The project is designed for maintainability, performance, and experimentation.

## Modular Workflow
1. **Data Layer**: Modular loaders, preprocessing, and splitting (`data/`, `util/data_utils.py`).
2. **Model Layer**: Unified model interface (`models/`), supports LSTM, RNN, MLP, LightGBM.
3. **Optimization Layer**: Meta-parameter search via PSO or Bayesian (`optimization/`).
4. **Training & Evaluation**: Modular training, cross-validation, and evaluation (`util/model_utils.py`).
5. **Ensembling**: Average, weighted, or stacking (meta-learner) via config.
6. **Experiment Tracking & Logging**: Utilities for logging, experiment tracking, and artifact management (`util/log_utils.py`, `util/experiment_tracker.py`).
7. **Plotting & Metrics**: Modular plotting and metrics utilities (`util/plot_utils.py`, `util/metrics.py`).
8. **Caching**: In-memory and disk caching for fast data reuse (`util/cache.py`).

## How to Run
```sh
python main.py
```

## Project Structure
- `main.py`: Orchestrates the modular pipeline.
- `config.py`: All tunable variables and workflow flags.
- `util/model_utils.py`: Pipeline logic, training, evaluation, ensembling, plotting.
- `util/cache.py`: In-memory/disk cache utility.
- `util/metrics.py`, `util/plot_utils.py`, `util/log_utils.py`, `util/experiment_tracker.py`: Utilities for metrics, plotting, logging, and experiment tracking.
- `optimization/`: Meta-parameter search (`meta_search.py`, `particle_swarm.py`, `bayesian_opt.py`).
- `models/`: Modular model definitions (LSTM, RNN, MLP, LightGBM).
- `data/`: Data loading, preprocessing, and splitting.

## Customization & Extensibility
- Edit `config.py` to change search spaces, workflow settings, or select optimization/ensembling/iterative stacking methods:
   - `META_OPT_METHOD = 'pso'` or `'bayesian'`
   - `CV_FOLDS = 1` (no CV) or `>1` (k-fold)
   - `ENSEMBLE_STRATEGY = 'average'`, `'weighted'`, or `'stacking'`
   - `ITERATIVE_STACKING = True` to enable meta-feature augmentation using previous predictions
- Add/remove meta-parameters in `optimization/meta_search.py`, `particle_swarm.py`, or `bayesian_opt.py`.
- Expand model search space or add new models in `models/` and `util/model_utils.py`.
- Use `util/cache.py` for caching intermediate results.

## Outputs
- `results_predictions.json`: Saved predictions and metrics (used for iterative stacking if enabled).
- `results_predictions_history.json`: History of predictions for meta-learning.
- Plots: Calibration, distributions, KL-divergence, etc.

## Requirements
- Python 3.8+
- TensorFlow, Keras, KerasTuner, Optuna, LightGBM, NumPy, pandas, matplotlib, scikit-learn

## License
MIT
