"""
Integration layer for performance-guided hyperparameter optimization.
"""
import numpy as np
import logging
from core.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class AdaptiveSearchSpace:
    """
    Dynamically adjust search space based on historical performance.
    """
    
    def __init__(self, performance_tracker=None):
        """
        Initialize adaptive search space manager.
        
        Args:
            performance_tracker: PerformanceTracker instance (creates new if None)
        """
        self.tracker = performance_tracker or PerformanceTracker()
    
    def get_pso_bounds(self, var_names, default_bounds, adapt=True):
        """
        Get PSO bounds, optionally adapted based on performance history.
        
        Args:
            var_names: List of parameter names
            default_bounds: Original bounds as list of (min, max) tuples
            adapt: If True, adjust bounds based on performance history
        
        Returns:
            Tuple of (lower_bounds, upper_bounds) as numpy arrays
        """
        if not adapt or len(self.tracker.history) < 10:
            # Not enough data to adapt, use defaults
            logger.info("[AdaptiveSearchSpace] Using default PSO bounds (insufficient history)")
            lower = np.array([b[0] for b in default_bounds])
            upper = np.array([b[1] for b in default_bounds])
            return lower, upper
        
        # Convert to dict format for suggestion
        bounds_dict = {name: bounds for name, bounds in zip(var_names, default_bounds)}
        suggested = self.tracker.suggest_search_bounds(bounds_dict)
        
        # Extract adapted bounds
        lower = []
        upper = []
        for name in var_names:
            if name in suggested:
                new_min, new_max = suggested[name]['suggested']
                lower.append(new_min)
                upper.append(new_max)
                logger.info(f"[AdaptiveSearchSpace] Adapted {name}: {suggested[name]['original']} -> "
                          f"{suggested[name]['suggested']} ({suggested[name]['reason']})")
            else:
                # Fall back to default
                idx = var_names.index(name)
                lower.append(default_bounds[idx][0])
                upper.append(default_bounds[idx][1])
        
        return np.array(lower), np.array(upper)
    
    def get_keras_hyperparameter_hints(self):
        """
        Get hints for KerasTuner based on performance analysis.
        
        Returns:
            Dict with parameter hints:
            {
                'param_name': {
                    'default': recommended_default_value,
                    'min': suggested_minimum,
                    'max': suggested_maximum,
                    'distribution': 'uniform' or 'log_uniform',
                    'confidence': 0.0-1.0 (how confident we are in this hint)
                }
            }
        """
        if len(self.tracker.history) < 10:
            logger.info("[AdaptiveSearchSpace] Insufficient history for KerasTuner hints")
            return {}
        
        analysis = self.tracker.analyze_parameter_patterns()
        hints = {}
        
        # Extract promising regions for keras params
        promising_keras = analysis['promising_regions'].get('keras_params', {})
        best_records = self.tracker.get_best_predictions(n=10)
        
        for param_name, param_range in promising_keras.items():
            # Collect values from best predictions
            values = []
            for record in best_records:
                if param_name in record.get('keras_params', {}):
                    values.append(record['keras_params'][param_name])
            
            if not values:
                continue
            
            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = param_range['min']
            max_val = param_range['max']
            
            # Determine if log distribution is appropriate
            # (if values span multiple orders of magnitude)
            distribution = 'log_uniform' if (max_val / max(min_val, 1e-10)) > 10 else 'uniform'
            
            # Confidence based on std/mean ratio (lower is more confident)
            confidence = max(0.0, min(1.0, 1.0 - (std_val / max(abs(mean_val), 1e-10))))
            
            hints[param_name] = {
                'default': float(mean_val),
                'min': float(min_val),
                'max': float(max_val),
                'distribution': distribution,
                'confidence': float(confidence),
                'reason': f'Based on {len(values)} best predictions'
            }
            
            logger.info(f"[AdaptiveSearchSpace] Hint for {param_name}: "
                       f"default={mean_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}], "
                       f"confidence={confidence:.2f}")
        
        return hints
    
    def create_informed_keras_tuner(self, hp, model_type='lstm'):
        """
        Create KerasTuner hyperparameter space informed by performance history.
        
        Args:
            hp: keras_tuner.HyperParameters instance
            model_type: Model type ('lstm', 'rnn', 'mlp')
        
        Returns:
            Configured HyperParameters instance
        """
        hints = self.get_keras_hyperparameter_hints()
        
        # Common hyperparameters with adaptive defaults
        if 'hidden_units' in hints:
            hint = hints['hidden_units']
            hp.Int('hidden_units', 
                   min_value=int(hint['min']), 
                   max_value=int(hint['max']), 
                   default=int(hint['default']),
                   step=16)
        else:
            # Default if no history
            hp.Int('hidden_units', min_value=32, max_value=256, default=64, step=16)
        
        if 'dropout_rate' in hints:
            hint = hints['dropout_rate']
            hp.Float('dropout_rate',
                    min_value=hint['min'],
                    max_value=hint['max'],
                    default=hint['default'],
                    step=0.1)
        else:
            hp.Float('dropout_rate', min_value=0.0, max_value=0.7, default=0.3, step=0.1)
        
        if 'learning_rate' in hints:
            hint = hints['learning_rate']
            if hint['distribution'] == 'log_uniform':
                hp.Float('learning_rate',
                        min_value=hint['min'],
                        max_value=hint['max'],
                        default=hint['default'],
                        sampling='log')
            else:
                hp.Float('learning_rate',
                        min_value=hint['min'],
                        max_value=hint['max'],
                        default=hint['default'])
        else:
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, default=1e-3, sampling='log')
        
        # Model-specific parameters
        if model_type in ['lstm', 'rnn']:
            if 'num_layers' in hints:
                hint = hints['num_layers']
                hp.Int('num_layers',
                      min_value=int(hint['min']),
                      max_value=int(hint['max']),
                      default=int(hint['default']))
            else:
                hp.Int('num_layers', min_value=1, max_value=3, default=2)
        
        logger.info(f"[AdaptiveSearchSpace] Created informed KerasTuner space with "
                   f"{len(hints)} adapted parameters")
        
        return hp
    
    def should_early_stop_search(self, current_iteration, max_iterations, 
                                 improvement_threshold=0.01, patience=5):
        """
        Determine if search should stop early based on performance plateau.
        
        Args:
            current_iteration: Current iteration number
            max_iterations: Maximum iterations planned
            improvement_threshold: Minimum improvement to consider progress
            patience: Number of iterations without improvement before stopping
        
        Returns:
            Tuple of (should_stop, reason)
        """
        if current_iteration < patience:
            return False, "Too early to stop"
        
        if len(self.tracker.history) < patience:
            return False, "Insufficient history"
        
        # Get recent predictions
        recent = self.tracker.history[-patience:]
        recent_matches = [r['quality'].get('matches', 0) for r in recent]
        
        # Check for improvement
        if not recent_matches:
            return False, "No match data"
        
        max_recent = max(recent_matches)
        
        # Get historical best before recent window
        historical = self.tracker.history[:-patience]
        if historical:
            historical_matches = [r['quality'].get('matches', 0) for r in historical]
            historical_best = max(historical_matches) if historical_matches else 0
            
            improvement = (max_recent - historical_best) / max(historical_best, 1)
            
            if improvement < improvement_threshold:
                return True, f"No significant improvement ({improvement:.2%}) in last {patience} iterations"
        
        return False, "Search is progressing"
    
    def get_parameter_importance_ranking(self):
        """
        Rank parameters by their impact on prediction quality.
        
        Returns:
            List of (param_name, importance_score, param_type) tuples, sorted by importance
        """
        if len(self.tracker.history) < 20:
            logger.warning("[AdaptiveSearchSpace] Insufficient data for importance ranking")
            return []
        
        # Get best and worst predictions
        best = self.tracker.get_best_predictions(n=10)
        worst = self.tracker.get_worst_predictions(n=10)
        
        importance_scores = {}
        
        for param_type in ['meta_params', 'keras_params']:
            # Collect all parameter names
            all_params = set()
            for record in best + worst:
                all_params.update(record.get(param_type, {}).keys())
            
            for param_name in all_params:
                # Get values from best and worst
                best_values = [r[param_type].get(param_name) 
                              for r in best if param_name in r.get(param_type, {})]
                worst_values = [r[param_type].get(param_name)
                               for r in worst if param_name in r.get(param_type, {})]
                
                # Filter numeric values
                best_values = [v for v in best_values if isinstance(v, (int, float, np.number))]
                worst_values = [v for v in worst_values if isinstance(v, (int, float, np.number))]
                
                if not best_values or not worst_values:
                    continue
                
                # Calculate separation between best and worst distributions
                best_mean = np.mean(best_values)
                worst_mean = np.mean(worst_values)
                best_std = np.std(best_values) if len(best_values) > 1 else 1.0
                worst_std = np.std(worst_values) if len(worst_values) > 1 else 1.0
                
                # Importance score: normalized distance between means
                pooled_std = np.sqrt((best_std**2 + worst_std**2) / 2)
                if pooled_std > 0:
                    importance = abs(best_mean - worst_mean) / pooled_std
                else:
                    importance = 0.0
                
                importance_scores[(param_name, param_type)] = importance
        
        # Sort by importance (descending)
        ranked = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        result = [(name, score, ptype) for (name, ptype), score in ranked]
        
        logger.info(f"[AdaptiveSearchSpace] Ranked {len(result)} parameters by importance")
        for name, score, ptype in result[:5]:
            logger.info(f"  {name} ({ptype}): {score:.3f}")
        
        return result
