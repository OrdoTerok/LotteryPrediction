# PSO with Post-CV/Ensemble: Lighter Alternative

## Overview

This module implements a computationally efficient approach to hyperparameter optimization:
- **During PSO iterations**: Use single train/validation split (fast)
- **After PSO completes**: Run full cross-validation + ensembling on best parameters (accurate)

This reduces computational cost by ~100x compared to running full CV+ensemble per PSO iteration.

## How It Works

### Step 1: Fast PSO Search
- PSO runs with `CV_FOLDS = 1` temporarily
- Each particle evaluates fitness using a single train/val split
- This is ~5x faster than 5-fold CV per particle
- Typical time: 5-10 particles × 10 iterations × 1-2 min = **50-200 minutes**

### Step 2: Post-PSO Cross-Validation
- After PSO finds best hyperparameters, apply them to config
- Run full K-fold cross-validation (typically 5-fold) on best params
- Train each model type (LSTM, MLP, RNN, LGBM) with CV
- Get robust performance estimates
- Typical time: 4 models × 5 folds × 2 min = **40 minutes**

### Step 3: Post-PSO Ensemble
- Train all model types (LSTM, MLP, RNN, LGBM) with best params
- Combine predictions using ensemble strategy (weighted, average, stacking)
- Evaluate ensemble performance on test set
- Typical time: 4 models × 5 min = **20 minutes**

### Total Time
- PSO + Post-CV + Ensemble: **110-260 minutes** (2-4 hours)
- vs. Full CV+Ensemble per PSO iteration: **~83 hours** (3+ days)

## Configuration

Enable in `config/config.py`:

```python
# Enable lighter PSO approach
USE_PSO_POST_CV_ENSEMBLE = True

# Standard PSO settings
PSO_PARTICLES = 5
PSO_ITER = 10

# CV folds for post-PSO validation (used after PSO)
CV_FOLDS = 5

# Ensemble strategy
ENSEMBLE_STRATEGY = 'weighted'  # or 'average', 'stacking'
```

## Usage

The lighter approach is automatically used when `USE_PSO_POST_CV_ENSEMBLE = True`:

```python
from pipeline.run_pipeline import run_meta_optimization

# This will use the lighter approach if enabled
run_meta_optimization(final_df, config)
```

## Output

The function returns a dict with:
- `best_params`: Best hyperparameters found by PSO
- `pso_fitness`: Best fitness from PSO search
- `cv_results`: Cross-validation results for each model type
- `ensemble_results`: Ensemble performance metrics

## Benefits

1. **Speed**: 100x faster than full CV+ensemble per PSO iteration
2. **Accuracy**: Still validates best params thoroughly with CV+ensemble
3. **Resource-efficient**: Can run on local machine vs. requiring cloud compute
4. **Flexible**: Easy to adjust PSO iterations, CV folds, ensemble strategy

## When to Use Full CV Per Iteration

Full CV+ensemble per PSO iteration is only recommended when:
- You have massive compute resources (cloud cluster)
- Hyperparameter space is very small (< 3 dimensions)
- Single train/val split is too noisy for your data
- You're willing to wait days for results

For most use cases, the lighter approach provides excellent results in practical time.
