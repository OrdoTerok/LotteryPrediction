# LotteryPrediction Project

## Overview
This project predicts lottery numbers using deep learning (LSTM/RNN) with advanced calibration, meta-optimization (PSO), and hyperparameter tuning (KerasTuner).

## Workflow
1. **Data Loading**: Loads and cleans Powerball data from multiple sources.
2. **Meta-Parameter Optimization (PSO)**: Particle Swarm Optimization tunes meta-parameters (label smoothing, uniform mix, temperature, early stopping, etc.).
3. **Model Hyperparameter Tuning (KerasTuner)**: Finds best model/training hyperparameters for the best meta-parameters.
4. **Evaluation**: Reports accuracy, calibration, entropy, KL, Brier score, and saves predictions/plots.

## How to Run
```sh
python main.py
```

## Key Files
- `main.py`: Orchestrates the workflow.
- `config.py`: All tunable variables.
- `util/model_utils.py`: Model training, tuning, evaluation, plotting.
- `particle_swarm.py`: PSO meta-optimization.
- `util/log_utils.py`: Logging utilities.

## Customization
- Edit `config.py` to change search spaces or workflow settings.
- Add/remove meta-parameters in `main.py` and `particle_swarm.py`.
- Expand model search space in `util/model_utils.py`.

## Outputs
- `results_predictions.json`: Saved predictions and metrics.
- `calibration_curve_first_ball.png`: Calibration plot.

## Requirements
- Python 3.8+
- TensorFlow, Keras, KerasTuner, NumPy, pandas, matplotlib, scikit-learn

## License
MIT
