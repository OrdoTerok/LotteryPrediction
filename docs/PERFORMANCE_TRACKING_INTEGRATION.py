"""
Example integration of performance tracking into the main pipeline.
"""

# In pipeline/run_pipeline.py, add these imports at the top:
from core.performance_tracker import PerformanceTracker
from optimization.adaptive_search import AdaptiveSearchSpace

# Initialize trackers early in run_pipeline():
def run_pipeline(config, best_pred=None):
    # ... existing code ...
    
    # Initialize performance tracking
    perf_tracker = PerformanceTracker()
    adaptive_search = AdaptiveSearchSpace(perf_tracker)
    
    # ... existing code ...

# When running meta-optimization (PSO), use adaptive bounds:
def run_meta_optimization(final_df, config, perf_tracker=None, adaptive_search=None):
    """
    Modified to use adaptive search bounds based on performance history.
    """
    # ... existing setup code ...
    
    # Get adaptive bounds if tracker available
    if adaptive_search is not None and getattr(config, 'USE_ADAPTIVE_SEARCH', True):
        # Original bounds
        default_bounds = [
            (0.0, 0.3),  # LABEL_SMOOTHING
            (1.0, 10.0),  # TEMP_MAX
            (5, 30),  # EARLY_STOPPING_PATIENCE
            (0.0, 2.0),  # OVERCOUNT_PENALTY_WEIGHT
        ]
        
        lower_bounds, upper_bounds = adaptive_search.get_pso_bounds(
            var_names, 
            default_bounds,
            adapt=True
        )
        
        bounds = (lower_bounds, upper_bounds)
        logger.info(f"[MetaOpt] Using adaptive PSO bounds: {bounds}")
    else:
        # Use default bounds
        bounds = (
            np.array([0.0, 1.0, 5, 0.0]),  # lower bounds
            np.array([0.3, 10.0, 30, 2.0])  # upper bounds
        )
    
    # Run PSO with adaptive bounds
    from optimization.particle_swarm import particle_swarm_optimize
    best_cost, best_pos = particle_swarm_optimize(
        var_names, 
        bounds, 
        final_df,
        n_particles=config.PSO_PARTICLES,
        n_iter=config.PSO_ITER
    )
    
    # ... rest of optimization ...

# When running KerasTuner, use informed hyperparameters:
def run_keras_tuner_with_adaptive_hints(model_type, X_train, y_train, adaptive_search=None):
    """
    Run KerasTuner with hints from performance history.
    """
    import keras_tuner as kt
    
    def build_model(hp):
        # Use adaptive search to inform hyperparameters
        if adaptive_search is not None:
            hp = adaptive_search.create_informed_keras_tuner(hp, model_type=model_type)
        else:
            # Default hyperparameters
            hp.Int('hidden_units', min_value=32, max_value=256, default=64, step=16)
            hp.Float('dropout_rate', min_value=0.0, max_value=0.7, default=0.3, step=0.1)
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, default=1e-3, sampling='log')
        
        # Build model with selected hyperparameters
        from models.model_factory import get_model
        model = get_model(
            model_type,
            input_shape=X_train.shape[1:],
            hidden_units=hp.get('hidden_units'),
            dropout_rate=hp.get('dropout_rate'),
            learning_rate=hp.get('learning_rate')
        )
        return model.model  # Return the Keras model
    
    # Create tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=30,
        directory='kt_dir',
        project_name=f'adaptive_{model_type}'
    )
    
    # Search
    tuner.search(X_train, y_train, epochs=30, validation_split=0.2, verbose=0)
    
    best_hp = tuner.get_best_hyperparameters()[0]
    return best_hp

# After making predictions, record performance:
def record_prediction_performance(perf_tracker, meta_params, keras_params, predictions, y_true):
    """
    Record prediction performance for future optimization.
    """
    # Calculate quality metrics
    matches = calculate_matches(predictions, y_true)  # Your existing function
    
    # Extract prediction quality
    quality = {
        'matches': matches,
        'first_five_accuracy': calculate_first_five_accuracy(predictions, y_true),
        'sixth_accuracy': calculate_sixth_accuracy(predictions, y_true),
        'total_loss': calculate_total_loss(predictions, y_true)
    }
    
    # Additional metrics from model evaluation
    metrics = {
        'first_five_loss': 0.0,  # Fill from model.evaluate()
        'sixth_loss': 0.0,
        'training_time': 0.0  # Track training duration
    }
    
    # Record
    perf_tracker.record_prediction(
        meta_params=meta_params,
        keras_params=keras_params,
        metrics=metrics,
        prediction_quality=quality
    )
    
    logger.info(f"[Pipeline] Recorded prediction with quality: {quality}")

# At the end of run_pipeline, before returning:
def finalize_performance_tracking(perf_tracker, adaptive_search):
    """
    Generate and save performance analysis.
    """
    # Export comprehensive analysis
    report = perf_tracker.export_analysis_report()
    
    # Get parameter importance ranking
    importance = adaptive_search.get_parameter_importance_ranking()
    
    logger.info("[Pipeline] Parameter importance ranking:")
    for param_name, score, param_type in importance[:10]:
        logger.info(f"  {param_name} ({param_type}): {score:.3f}")
    
    # Check if we should adjust search strategy
    if len(perf_tracker.history) > 50:
        worst = perf_tracker.get_worst_predictions(n=10)
        logger.info(f"[Pipeline] Worst predictions had {[w['quality']['matches'] for w in worst]} matches")
        logger.info("[Pipeline] Consider adjusting search bounds based on analysis report")

# Example of early stopping for PSO:
def pso_with_early_stopping(var_names, bounds, final_df, n_iter, adaptive_search):
    """
    PSO optimization with adaptive early stopping.
    """
    for iteration in range(n_iter):
        # Run PSO iteration
        # ... your PSO code ...
        
        # Check if should stop early
        should_stop, reason = adaptive_search.should_early_stop_search(
            current_iteration=iteration,
            max_iterations=n_iter,
            improvement_threshold=0.01,
            patience=5
        )
        
        if should_stop:
            logger.info(f"[PSO] Early stopping at iteration {iteration}: {reason}")
            break
    
    return best_cost, best_pos
