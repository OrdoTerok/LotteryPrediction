# Performance Tracking and Adaptive Search

## Overview

This system tracks the worst and best predictions along with their associated hyperparameters (both meta-parameters from PSO/Bayesian optimization and KerasTuner hyperparameters) to intelligently shape future parameter search spaces.

## Key Components

### 1. PerformanceTracker (`core/performance_tracker.py`)
Stores and analyzes prediction history with associated hyperparameters.

**Features:**
- Records predictions with meta-params, keras-params, and quality metrics
- Identifies worst/best predictions
- Analyzes parameter patterns
- Suggests updated search bounds
- Exports comprehensive analysis reports

### 2. AdaptiveSearchSpace (`optimization/adaptive_search.py`)
Uses performance history to adapt search strategies.

**Features:**
- Adjusts PSO bounds based on worst/best regions
- Provides informed defaults for KerasTuner
- Ranks parameters by importance
- Implements early stopping logic

### 3. Visualization Tools (`visualization/performance_viz.py`)
Generates plots to understand performance patterns.

**Visualizations:**
- Performance history over time
- Parameter distributions (best vs worst)
- Parameter importance ranking
- Search space evolution

## How It Works

### 1. Recording Performance

After each prediction:
```python
from core.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# After making prediction
tracker.record_prediction(
    meta_params={
        'LABEL_SMOOTHING': 0.1,
        'TEMP_MAX': 5.0,
        'EARLY_STOPPING_PATIENCE': 10,
        'OVERCOUNT_PENALTY_WEIGHT': 0.5
    },
    keras_params={
        'hidden_units': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'num_layers': 2
    },
    metrics={
        'first_five_loss': 3.45,
        'sixth_loss': 2.89,
        'training_time': 125.3
    },
    prediction_quality={
        'matches': 2,  # Number of correct predictions
        'first_five_accuracy': 0.42,
        'sixth_accuracy': 0.18,
        'total_loss': 6.34
    }
)
```

### 2. Analyzing Patterns

The system identifies:
- **Avoid Regions**: Parameter ranges that led to worst predictions
- **Promising Regions**: Parameter ranges that led to best predictions
- **Parameter Importance**: Which parameters have most impact

```python
from optimization.adaptive_search import AdaptiveSearchSpace

adaptive_search = AdaptiveSearchSpace(tracker)

# Get analysis
analysis = tracker.analyze_parameter_patterns(worst_n=20, best_n=20)

# Example output:
# {
#   'avoid_regions': {
#     'meta_params': {
#       'TEMP_MAX': {'min': 8.5, 'max': 10.0, 'reason': 'No overlap with best'}
#     }
#   },
#   'promising_regions': {
#     'keras_params': {
#       'hidden_units': {'min': 96, 'max': 160, 'reason': 'Distinct from worst'}
#     }
#   }
# }
```

### 3. Adapting PSO Bounds

```python
# Original bounds
default_bounds = [
    (0.0, 0.3),  # LABEL_SMOOTHING
    (1.0, 10.0),  # TEMP_MAX
    (5, 30),      # EARLY_STOPPING_PATIENCE
    (0.0, 2.0),   # OVERCOUNT_PENALTY_WEIGHT
]

# Get adapted bounds
lower, upper = adaptive_search.get_pso_bounds(
    var_names=['LABEL_SMOOTHING', 'TEMP_MAX', 'EARLY_STOPPING_PATIENCE', 'OVERCOUNT_PENALTY_WEIGHT'],
    default_bounds=default_bounds,
    adapt=True
)

# If worst predictions had TEMP_MAX in range [8.5, 10.0],
# the adapted bounds might shift to [1.0, 8.0] to avoid that region
```

### 4. Informing KerasTuner

```python
import keras_tuner as kt

def build_model(hp):
    # Use adaptive hints
    hp = adaptive_search.create_informed_keras_tuner(hp, model_type='lstm')
    
    # This automatically sets:
    # - Default values based on best predictions
    # - Min/max ranges avoiding worst regions
    # - Appropriate sampling distribution (uniform vs log)
    
    hidden_units = hp.get('hidden_units')
    dropout_rate = hp.get('dropout_rate')
    learning_rate = hp.get('learning_rate')
    
    # Build and return model
    # ...
```

### 5. Parameter Importance

```python
# Rank parameters by impact
importance = adaptive_search.get_parameter_importance_ranking()

# Output:
# [
#   ('learning_rate', 2.45, 'keras_params'),      # Highest impact
#   ('TEMP_MAX', 1.89, 'meta_params'),
#   ('hidden_units', 1.67, 'keras_params'),
#   ('dropout_rate', 1.23, 'keras_params'),
#   # ... etc
# ]

# Focus optimization on top parameters
```

### 6. Early Stopping

```python
for iteration in range(max_iterations):
    # Run optimization iteration
    # ...
    
    # Check if should stop
    should_stop, reason = adaptive_search.should_early_stop_search(
        current_iteration=iteration,
        max_iterations=max_iterations,
        improvement_threshold=0.01,
        patience=5
    )
    
    if should_stop:
        logger.info(f"Early stopping: {reason}")
        break
```

## Configuration

Add to `config/config.py`:

```python
# Enable performance tracking
ENABLE_PERFORMANCE_TRACKING = True

# Use adaptive search bounds
USE_ADAPTIVE_SEARCH = True

# Minimum history before adapting (default: 10)
MIN_HISTORY_FOR_ADAPTATION = 10

# Generate visualizations
GENERATE_PERFORMANCE_VIZ = True
```

## Integration Example

```python
# In pipeline/run_pipeline.py

from core.performance_tracker import PerformanceTracker
from optimization.adaptive_search import AdaptiveSearchSpace
from visualization.performance_viz import generate_all_visualizations

def run_pipeline(config, best_pred=None):
    # Initialize tracking
    tracker = PerformanceTracker()
    adaptive_search = AdaptiveSearchSpace(tracker)
    
    # ... existing setup ...
    
    # Run meta-optimization with adaptive bounds
    if getattr(config, 'USE_ADAPTIVE_SEARCH', True):
        run_meta_optimization_adaptive(final_df, config, adaptive_search)
    else:
        run_meta_optimization(final_df, config)
    
    # ... train models ...
    
    # Make prediction
    prediction = make_final_prediction(models, X_test)
    
    # Record performance
    if getattr(config, 'ENABLE_PERFORMANCE_TRACKING', True):
        tracker.record_prediction(
            meta_params=get_current_meta_params(config),
            keras_params=get_keras_hyperparams(best_model),
            metrics=evaluation_metrics,
            prediction_quality=calculate_quality(prediction, y_true)
        )
    
    # Generate visualizations
    if getattr(config, 'GENERATE_PERFORMANCE_VIZ', True):
        generate_all_visualizations(tracker, adaptive_search, 
                                    output_dir='experiments/viz')
    
    # Export analysis
    tracker.export_analysis_report('experiments/performance_analysis.json')
    
    return prediction
```

## Output Files

### performance_history.json
Complete history of all predictions:
```json
[
  {
    "timestamp": "2025-11-04T10:30:45",
    "meta_params": { "LABEL_SMOOTHING": 0.1, ... },
    "keras_params": { "hidden_units": 128, ... },
    "quality": { "matches": 2, "total_loss": 6.34 },
    "is_worst": false,
    "is_best": false
  },
  ...
]
```

### performance_analysis.json
Comprehensive analysis:
```json
{
  "generated_at": "2025-11-04T11:45:00",
  "total_records": 145,
  "analysis": {
    "worst_param_ranges": { ... },
    "best_param_ranges": { ... },
    "avoid_regions": { ... },
    "promising_regions": { ... }
  },
  "recommendations": {
    "focus_on": ["learning_rate", "hidden_units"],
    "avoid": ["TEMP_MAX ranges 8.5-10.0"]
  }
}
```

### Visualizations (experiments/viz/)
- `performance_history.png` - Quality over time
- `param_importance.png` - Parameter importance ranking
- `dist_{param}.png` - Parameter distribution (best vs worst)
- `evolution_{param}.png` - Parameter evolution over time

## Benefits

1. **Faster Convergence**: Avoid known bad parameter regions
2. **Better Defaults**: Start with values that worked well historically
3. **Focused Search**: Concentrate on high-impact parameters
4. **Early Stopping**: Stop when no improvement is detected
5. **Interpretability**: Understand which parameters matter most

## Best Practices

1. **Initial Phase**: Run 10-20 predictions with default bounds to build history
2. **Adaptation Phase**: Enable adaptive search after sufficient history
3. **Regular Review**: Check analysis reports to understand patterns
4. **Visualization**: Use plots to verify adaptations make sense
5. **Backup History**: Keep `performance_history.json` backed up

## Advanced Usage

### Custom Metrics
```python
tracker.record_prediction(
    meta_params={...},
    keras_params={...},
    metrics={...},
    prediction_quality={
        'matches': 2,
        'custom_score': calculate_custom_score(pred, true),
        'diversity_score': calculate_diversity(pred)
    }
)

# Analyze by custom metric
worst = tracker.get_worst_predictions(n=10, metric='custom_score')
```

### Multi-Objective Optimization
```python
# Balance multiple objectives
analysis = tracker.analyze_parameter_patterns()

# Find params that balance accuracy AND speed
fast_and_accurate = [
    r for r in tracker.history
    if r['quality']['matches'] >= 2 
    and r['metrics']['training_time'] < 100
]
```

### Transfer Learning
```python
# Load history from previous project
tracker = PerformanceTracker(history_file='previous_project/history.json')

# Use as starting point for new optimization
adaptive_search = AdaptiveSearchSpace(tracker)
bounds = adaptive_search.get_pso_bounds(var_names, default_bounds)
```

## Troubleshooting

**Q: Adaptive bounds are too restrictive**
- Increase `shrink_factor` in `suggest_search_bounds()`
- Verify sufficient diversity in history (check visualizations)

**Q: No adaptation happening**
- Check `len(tracker.history) >= MIN_HISTORY_FOR_ADAPTATION`
- Ensure parameters are being recorded correctly

**Q: Importance scores all similar**
- May indicate all parameters equally important
- Or insufficient variation in parameter values
- Try wider default bounds to explore more

## Future Enhancements

- Bayesian optimization integration
- Multi-fidelity optimization (early stopping during training)
- Automated A/B testing of search strategies
- Real-time adaptation during PSO iterations
- Ensemble of search strategies based on confidence
