"""
PSO with Post-Optimization CV and Ensembling
---------------------------------------------
Lighter alternative: run PSO with single train/val splits, then apply
full cross-validation and ensembling only to the best hyperparameters found.

This dramatically reduces computational cost while still finding good hyperparameters.
"""
import numpy as np
import logging
from optimization.particle_swarm import particle_swarm_optimize
from core.log_utils import get_logger

logger = get_logger()

def run_pso_with_post_cv_ensemble(var_names, bounds, final_df, config, 
                                   pso_particles=5, pso_iter=10, 
                                   post_cv_folds=5, use_ensemble=True):
    """
    Run PSO optimization with lightweight fitness evaluation, then apply
    full cross-validation and ensembling to the best parameters.
    
    Args:
        var_names: List of parameter names to optimize.
        bounds: List of (low, high) tuples for each parameter.
        final_df: Tuple of (train_df, test_df).
        config: Configuration object.
        pso_particles: Number of PSO particles (default 5).
        pso_iter: Number of PSO iterations (default 10).
        post_cv_folds: Number of CV folds to run after PSO (default 5).
        use_ensemble: Whether to run ensemble after PSO (default True).
        
    Returns:
        dict with keys:
            - 'best_params': List of best parameter values from PSO
            - 'pso_fitness': Best fitness from PSO
            - 'cv_results': Cross-validation results on best params (if post_cv_folds > 1)
            - 'ensemble_results': Ensemble results on best params (if use_ensemble)
    """
    logger.info("[PSO-Post-CV] Step 1: Running PSO with single train/val split...")
    
    # Step 1: Save original CV setting and temporarily disable CV during PSO
    original_cv_folds = getattr(config, 'CV_FOLDS', 1)
    setattr(config, 'CV_FOLDS', 1)  # Force single train/val during PSO
    
    try:
        # Run PSO with lightweight fitness (single split)
        best_params = particle_swarm_optimize(
            var_names, 
            bounds, 
            final_df, 
            n_particles=pso_particles, 
            n_iter=pso_iter,
            cv=1  # Explicitly pass cv=1 to ensure single split
        )
        
        if best_params is None:
            logger.error("[PSO-Post-CV] PSO failed to find best parameters.")
            return None
            
        logger.info(f"[PSO-Post-CV] Step 1 complete. Best PSO params: {dict(zip(var_names, best_params))}")
        
        # Apply best params to config
        for i, name in enumerate(var_names):
            setattr(config, name, type(getattr(config, name))(best_params[i]))
        
        result = {
            'best_params': best_params,
            'pso_fitness': None,  # Will be filled below
        }
        
        # Step 2: Run full cross-validation on best parameters
        if post_cv_folds > 1:
            logger.info(f"[PSO-Post-CV] Step 2: Running {post_cv_folds}-fold CV on best parameters...")
            cv_results = run_cv_on_best_params(final_df, config, post_cv_folds)
            result['cv_results'] = cv_results
            logger.info(f"[PSO-Post-CV] Step 2 complete. CV results: {cv_results}")
        else:
            logger.info("[PSO-Post-CV] Step 2 skipped (post_cv_folds <= 1).")
            result['cv_results'] = None
        
        # Step 3: Run ensemble on best parameters
        if use_ensemble:
            logger.info("[PSO-Post-CV] Step 3: Running ensemble on best parameters...")
            ensemble_results = run_ensemble_on_best_params(final_df, config)
            result['ensemble_results'] = ensemble_results
            logger.info(f"[PSO-Post-CV] Step 3 complete. Ensemble results: {ensemble_results}")
        else:
            logger.info("[PSO-Post-CV] Step 3 skipped (use_ensemble=False).")
            result['ensemble_results'] = None
            
        return result
        
    finally:
        # Restore original CV setting
        setattr(config, 'CV_FOLDS', original_cv_folds)
        logger.info(f"[PSO-Post-CV] Restored CV_FOLDS to {original_cv_folds}")


def run_cv_on_best_params(final_df, config, cv_folds=5):
    """
    Run cross-validation on the best parameters found by PSO.
    
    Args:
        final_df: Tuple of (train_df, test_df).
        config: Configuration object with best parameters already set.
        cv_folds: Number of CV folds.
        
    Returns:
        dict with CV results per model type
    """
    from data.preprocessing import prepare_data_for_lstm
    from models.model_factory import get_model
    
    train_df, test_df = final_df
    look_back_window = getattr(config, 'LOOK_BACK_WINDOW', 10)
    X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
    
    if X_train.size == 0:
        logger.warning("[PSO-Post-CV] Not enough data for CV.")
        return None
    
    # Run CV for each model type
    model_types = ['lstm', 'mlp', 'rnn', 'lgbm']
    cv_results = {}
    
    for model_type in model_types:
        logger.info(f"[PSO-Post-CV] Running {cv_folds}-fold CV for {model_type.upper()}...")
        try:
            input_shape = X_train.shape[1:]
            model = get_model(model_type, input_shape=input_shape)
            
            # Run cross-validation
            fold_results = model.cross_validate(X_train, y_train, cv=cv_folds, epochs=5, batch_size=32, verbose=0)
            
            # Extract mean metrics
            cv_results[model_type] = {
                'fold_results': fold_results,
                'mean_loss': np.mean([extract_loss(r) for r in fold_results]),
                'std_loss': np.std([extract_loss(r) for r in fold_results])
            }
            logger.info(f"[PSO-Post-CV] {model_type.upper()} CV complete: mean_loss={cv_results[model_type]['mean_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"[PSO-Post-CV] Error during CV for {model_type}: {e}")
            cv_results[model_type] = {'error': str(e)}
    
    return cv_results


def run_ensemble_on_best_params(final_df, config):
    """
    Run ensemble prediction on the best parameters found by PSO.
    
    Args:
        final_df: Tuple of (train_df, test_df).
        config: Configuration object with best parameters already set.
        
    Returns:
        dict with ensemble results
    """
    from data.preprocessing import prepare_data_for_lstm
    from models.model_factory import get_model
    from ensemble.ensemble_predict import ensemble_predict
    
    train_df, test_df = final_df
    look_back_window = getattr(config, 'LOOK_BACK_WINDOW', 10)
    X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
    X_test, y_test = prepare_data_for_lstm(test_df, look_back=look_back_window)
    
    if X_train.size == 0 or X_test.size == 0:
        logger.warning("[PSO-Post-CV] Not enough data for ensemble.")
        return None
    
    # Train all model types
    model_types = ['lstm', 'mlp', 'rnn', 'lgbm']
    models = {}
    
    for model_type in model_types:
        logger.info(f"[PSO-Post-CV] Training {model_type.upper()} for ensemble...")
        try:
            input_shape = X_train.shape[1:]
            model = get_model(model_type, input_shape=input_shape)
            # Train model directly using its fit method
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
            models[model_type] = model
            logger.info(f"[PSO-Post-CV] {model_type.upper()} training complete.")
        except Exception as e:
            logger.error(f"[PSO-Post-CV] Error training {model_type}: {e}")
    
    if len(models) == 0:
        logger.error("[PSO-Post-CV] No models trained successfully.")
        return None
    
    # Run ensemble prediction
    logger.info(f"[PSO-Post-CV] Running ensemble prediction with {len(models)} models...")
    try:
        ensemble_strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'weighted')
        ensemble_result = ensemble_predict(
            list(models.values()),
            X_test,
            strategy=ensemble_strategy
        )
        
        # Evaluate ensemble
        from sklearn.metrics import log_loss
        y_test_first, y_test_sixth = y_test
        
        # Convert one-hot to indices if needed
        if y_test_first.ndim == 3:
            y_test_first = np.argmax(y_test_first, axis=-1)
        if y_test_sixth.ndim == 3:
            y_test_sixth = np.argmax(y_test_sixth, axis=-1)
        
        first_five_pred, sixth_pred = ensemble_result
        
        # Calculate losses
        losses = []
        for i in range(5):
            losses.append(log_loss(y_test_first[:, i], first_five_pred[:, i, :], labels=np.arange(69)))
        losses.append(log_loss(y_test_sixth[:, 0], sixth_pred[:, 0, :], labels=np.arange(26)))
        
        mean_loss = np.mean(losses)
        
        logger.info(f"[PSO-Post-CV] Ensemble complete: mean_loss={mean_loss:.4f}")
        
        return {
            'strategy': ensemble_strategy,
            'num_models': len(models),
            'losses': losses,
            'mean_loss': mean_loss
        }
        
    except Exception as e:
        logger.error(f"[PSO-Post-CV] Error during ensemble prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def extract_loss(fold_result):
    """
    Extract loss value from cross-validation fold result.
    
    Args:
        fold_result: Result dict from a single CV fold.
        
    Returns:
        float: Loss value
    """
    # Handle nested dict structures
    val = fold_result
    if isinstance(val, dict) and 'eval' in val:
        val = val['eval']
    if isinstance(val, dict) and 'results' in val:
        val = val['results']
    if isinstance(val, dict) and 'losses' in val:
        val = val['losses']
    if isinstance(val, dict):
        val = list(val.values())[0]
    if isinstance(val, (list, tuple, np.ndarray)):
        val = np.mean(val)
    return float(val)
