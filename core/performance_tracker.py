"""
Performance tracking for worst/best predictions to guide hyperparameter search.
"""
import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Track prediction performance with associated hyperparameters to guide search space optimization.
    """
    
    def __init__(self, history_file='experiments/performance_history.json', max_history=1000):
        """
        Initialize performance tracker.
        
        Args:
            history_file: Path to JSON file storing performance history
            max_history: Maximum number of records to keep in memory
        """
        self.history_file = history_file
        self.max_history = max_history
        self.history = []
        self._load_history()
        
    def _load_history(self):
        """Load existing performance history from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"[PerformanceTracker] Loaded {len(self.history)} historical records")
            except Exception as e:
                logger.error(f"[PerformanceTracker] Failed to load history: {e}")
                self.history = []
    
    def _save_history(self):
        """Save performance history to disk."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        try:
            # Keep only the most recent max_history records
            history_to_save = self.history[-self.max_history:]
            with open(self.history_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
            logger.info(f"[PerformanceTracker] Saved {len(history_to_save)} records to {self.history_file}")
        except Exception as e:
            logger.error(f"[PerformanceTracker] Failed to save history: {e}")
    
    def record_prediction(self, meta_params, keras_params, metrics, prediction_quality):
        """
        Record a prediction with its hyperparameters and performance metrics.
        
        Args:
            meta_params: Dict of meta-parameters (PSO/Bayesian optimized)
            keras_params: Dict of KerasTuner hyperparameters
            metrics: Dict of performance metrics (loss, accuracy, etc.)
            prediction_quality: Dict with quality indicators:
                - 'matches': number of correct predictions
                - 'first_five_accuracy': accuracy for first 5 balls
                - 'sixth_accuracy': accuracy for powerball
                - 'total_loss': combined loss value
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'meta_params': self._sanitize_params(meta_params),
            'keras_params': self._sanitize_params(keras_params),
            'metrics': self._sanitize_params(metrics),
            'quality': self._sanitize_params(prediction_quality),
            'is_worst': False,  # Will be updated by analysis
            'is_best': False
        }
        
        self.history.append(record)
        self._save_history()
        logger.info(f"[PerformanceTracker] Recorded prediction with {prediction_quality.get('matches', 0)} matches")
        
        return record
    
    def _sanitize_params(self, params):
        """Convert numpy types to Python types for JSON serialization."""
        if params is None:
            return {}
        
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, (np.integer, np.int32, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, np.ndarray):
                sanitized[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [self._sanitize_value(v) for v in value]
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_value(self, value):
        """Sanitize a single value."""
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    def get_worst_predictions(self, n=20, metric='matches'):
        """
        Get N worst predictions based on a metric.
        
        Args:
            n: Number of worst predictions to return
            metric: Metric to use for ranking ('matches', 'total_loss', 'first_five_accuracy', etc.)
        
        Returns:
            List of worst prediction records, sorted by metric
        """
        if not self.history:
            return []
        
        # Filter records that have the metric
        valid_records = [r for r in self.history if metric in r.get('quality', {})]
        
        if not valid_records:
            logger.warning(f"[PerformanceTracker] No records found with metric '{metric}'")
            return []
        
        # Sort by metric (lower matches = worse, higher loss = worse)
        if metric in ['matches', 'first_five_accuracy', 'sixth_accuracy']:
            # Lower is worse for these metrics
            sorted_records = sorted(valid_records, key=lambda r: r['quality'][metric])
        else:
            # Higher is worse for loss metrics
            sorted_records = sorted(valid_records, key=lambda r: r['quality'][metric], reverse=True)
        
        worst = sorted_records[:n]
        
        # Mark as worst
        for record in worst:
            record['is_worst'] = True
        
        logger.info(f"[PerformanceTracker] Found {len(worst)} worst predictions by {metric}")
        return worst
    
    def get_best_predictions(self, n=20, metric='matches'):
        """
        Get N best predictions based on a metric.
        
        Args:
            n: Number of best predictions to return
            metric: Metric to use for ranking
        
        Returns:
            List of best prediction records, sorted by metric
        """
        if not self.history:
            return []
        
        valid_records = [r for r in self.history if metric in r.get('quality', {})]
        
        if not valid_records:
            return []
        
        # Sort by metric (higher matches = better, lower loss = better)
        if metric in ['matches', 'first_five_accuracy', 'sixth_accuracy']:
            sorted_records = sorted(valid_records, key=lambda r: r['quality'][metric], reverse=True)
        else:
            sorted_records = sorted(valid_records, key=lambda r: r['quality'][metric])
        
        best = sorted_records[:n]
        
        for record in best:
            record['is_best'] = True
        
        logger.info(f"[PerformanceTracker] Found {len(best)} best predictions by {metric}")
        return best
    
    def analyze_parameter_patterns(self, worst_n=20, best_n=20):
        """
        Analyze parameter patterns in worst vs best predictions.
        
        Returns:
            Dict with analysis results including:
            - 'worst_param_ranges': Parameter ranges that led to worst predictions
            - 'best_param_ranges': Parameter ranges that led to best predictions
            - 'avoid_regions': Parameter regions to avoid in future searches
            - 'promising_regions': Parameter regions to focus on
        """
        worst = self.get_worst_predictions(n=worst_n, metric='matches')
        best = self.get_best_predictions(n=best_n, metric='matches')
        
        analysis = {
            'worst_param_ranges': self._compute_param_ranges(worst),
            'best_param_ranges': self._compute_param_ranges(best),
            'avoid_regions': {},
            'promising_regions': {}
        }
        
        # Identify regions to avoid (worst ranges that don't overlap with best)
        for param_type in ['meta_params', 'keras_params']:
            if param_type not in analysis['worst_param_ranges']:
                continue
                
            analysis['avoid_regions'][param_type] = {}
            analysis['promising_regions'][param_type] = {}
            
            for param_name, worst_range in analysis['worst_param_ranges'][param_type].items():
                best_range = analysis['best_param_ranges'].get(param_type, {}).get(param_name)
                
                if best_range is None:
                    continue
                
                # If worst range is mostly outside best range, mark it to avoid
                worst_min, worst_max = worst_range['min'], worst_range['max']
                best_min, best_max = best_range['min'], best_range['max']
                
                # Check for non-overlapping or minimal overlap
                if worst_max < best_min or worst_min > best_max:
                    # No overlap - definitely avoid worst range
                    analysis['avoid_regions'][param_type][param_name] = {
                        'min': worst_min,
                        'max': worst_max,
                        'reason': 'No overlap with best predictions'
                    }
                
                # Mark best range as promising if it's distinct from worst
                if best_min > worst_max or best_max < worst_min:
                    analysis['promising_regions'][param_type][param_name] = {
                        'min': best_min,
                        'max': best_max,
                        'reason': 'Distinct from worst predictions'
                    }
        
        logger.info(f"[PerformanceTracker] Analysis complete: "
                   f"{len(analysis['avoid_regions'].get('meta_params', {}))} meta-params to avoid, "
                   f"{len(analysis['avoid_regions'].get('keras_params', {}))} keras params to avoid")
        
        return analysis
    
    def _compute_param_ranges(self, records):
        """
        Compute min/max/mean/std for each parameter across records.
        
        Returns:
            Dict with parameter statistics
        """
        if not records:
            return {}
        
        param_values = defaultdict(lambda: defaultdict(list))
        
        for record in records:
            for param_type in ['meta_params', 'keras_params']:
                if param_type not in record:
                    continue
                for param_name, param_value in record[param_type].items():
                    if isinstance(param_value, (int, float, np.number)):
                        param_values[param_type][param_name].append(float(param_value))
        
        ranges = {}
        for param_type, params in param_values.items():
            ranges[param_type] = {}
            for param_name, values in params.items():
                if not values:
                    continue
                ranges[param_type][param_name] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values)
                }
        
        return ranges
    
    def suggest_search_bounds(self, original_bounds, shrink_factor=0.7, expand_factor=1.3):
        """
        Suggest updated search bounds based on performance analysis.
        
        Args:
            original_bounds: Dict of parameter bounds {'param_name': (min, max), ...}
            shrink_factor: Factor to shrink bounds when avoiding bad regions (0-1)
            expand_factor: Factor to expand bounds for promising regions (>1)
        
        Returns:
            Dict with suggested bounds and reasoning
        """
        analysis = self.analyze_parameter_patterns()
        suggested_bounds = {}
        
        for param_name, (orig_min, orig_max) in original_bounds.items():
            # Check if this param appears in avoid or promising regions
            avoid_meta = analysis['avoid_regions'].get('meta_params', {}).get(param_name)
            avoid_keras = analysis['avoid_regions'].get('keras_params', {}).get(param_name)
            promising_meta = analysis['promising_regions'].get('meta_params', {}).get(param_name)
            promising_keras = analysis['promising_regions'].get('keras_params', {}).get(param_name)
            
            avoid = avoid_meta or avoid_keras
            promising = promising_meta or promising_keras
            
            new_min, new_max = orig_min, orig_max
            reason = "No change (insufficient data)"
            
            if promising:
                # Focus search on promising region
                prom_min, prom_max = promising['min'], promising['max']
                # Expand promising region slightly
                range_size = prom_max - prom_min
                new_min = max(orig_min, prom_min - range_size * (expand_factor - 1) / 2)
                new_max = min(orig_max, prom_max + range_size * (expand_factor - 1) / 2)
                reason = f"Focused on promising region: {promising['reason']}"
            
            elif avoid:
                # Shrink away from avoid region
                avoid_min, avoid_max = avoid['min'], avoid['max']
                
                # If avoid region is in lower half, shift bounds up
                if avoid_max < (orig_min + orig_max) / 2:
                    new_min = max(orig_min, avoid_max)
                    reason = f"Avoiding lower region: {avoid['reason']}"
                # If avoid region is in upper half, shift bounds down
                elif avoid_min > (orig_min + orig_max) / 2:
                    new_max = min(orig_max, avoid_min)
                    reason = f"Avoiding upper region: {avoid['reason']}"
                # If avoid region is in middle, split the search space
                else:
                    # Use the larger remaining region
                    lower_size = avoid_min - orig_min
                    upper_size = orig_max - avoid_max
                    if lower_size > upper_size:
                        new_max = avoid_min
                        reason = "Avoiding middle region (using lower half)"
                    else:
                        new_min = avoid_max
                        reason = "Avoiding middle region (using upper half)"
            
            suggested_bounds[param_name] = {
                'original': (orig_min, orig_max),
                'suggested': (new_min, new_max),
                'reason': reason
            }
        
        logger.info(f"[PerformanceTracker] Generated suggested bounds for {len(suggested_bounds)} parameters")
        return suggested_bounds
    
    def export_analysis_report(self, output_file='experiments/performance_analysis.json'):
        """
        Export comprehensive analysis report to file.
        """
        analysis = self.analyze_parameter_patterns()
        worst = self.get_worst_predictions(n=20)
        best = self.get_best_predictions(n=20)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(self.history),
            'analysis': analysis,
            'worst_predictions': worst,
            'best_predictions': best,
            'recommendations': {
                'focus_on': list(analysis['promising_regions'].get('meta_params', {}).keys()) + 
                           list(analysis['promising_regions'].get('keras_params', {}).keys()),
                'avoid': list(analysis['avoid_regions'].get('meta_params', {}).keys()) +
                        list(analysis['avoid_regions'].get('keras_params', {}).keys())
            }
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[PerformanceTracker] Exported analysis report to {output_file}")
        return report
