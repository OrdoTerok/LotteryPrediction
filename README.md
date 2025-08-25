# LotteryPrediction Project

## Overview
This project predicts lottery numbers using deep learning (LSTM/RNN) with advanced calibration, meta-optimization (PSO or Bayesian), cross-validation, and flexible ensembling (average, weighted, stacking).

## Workflow
1. **Data Loading**: Loads and cleans Powerball data from multiple sources.
2. **Meta-Parameter Optimization**: Selectable via config (`META_OPT_METHOD`):
   - Particle Swarm Optimization (PSO)
   - Bayesian Optimization (Optuna)
3. **Model Hyperparameter Tuning (KerasTuner)**: Finds best model/training hyperparameters for the best meta-parameters.
4. **Cross-Validation**: Robust evaluation with k-fold CV (set `CV_FOLDS` in config).
5. **Ensembling**: Selectable via config (`ENSEMBLE_STRATEGY`):
   - Average, Weighted, or Stacking (meta-learner)
6. **Iterative Stacking (Meta-Features)**: If `ITERATIVE_STACKING` is enabled in `config.py`, predictions from the previous run are loaded and appended as meta-features for the next run, enabling iterative meta-learning.
7. **Evaluation**: Reports accuracy, calibration, entropy, KL, Brier score, and saves predictions/plots.

## How to Run
```sh
python main.py
```

## Key Files
- `main.py`: Orchestrates the workflow.
- `config.py`: All tunable variables and workflow flags.
- `util/model_utils.py`: Model training, tuning, evaluation, plotting, ensembling.
- `particle_swarm.py`: PSO meta-optimization.
- `bayesian_opt.py`: Bayesian meta-optimization (Optuna).
- `util/log_utils.py`: Logging utilities.

## Customization
- Edit `config.py` to change search spaces, workflow settings, or select optimization/ensembling/iterative stacking methods:
   - `META_OPT_METHOD = 'pso'` or `'bayesian'`
   - `CV_FOLDS = 1` (no CV) or `>1` (k-fold)
   - `ENSEMBLE_STRATEGY = 'average'`, `'weighted'`, or `'stacking'`
   - `ITERATIVE_STACKING = True` to enable meta-feature augmentation using previous predictions
- Add/remove meta-parameters in `main.py`, `particle_swarm.py`, or `bayesian_opt.py`.
- Expand model search space in `util/model_utils.py`.

## Outputs
- `results_predictions.json`: Saved predictions and metrics (used for iterative stacking if enabled).
- `calibration_curve_first_ball.png`: Calibration plot.

## Requirements
- Python 3.8+
- TensorFlow, Keras, KerasTuner, Optuna, NumPy, pandas, matplotlib, scikit-learn

## License
MIT
