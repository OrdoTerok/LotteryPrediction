"""
Visualization tools for performance tracking analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
from core.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


def plot_performance_history(tracker, metric='matches', save_path='experiments/performance_history.png'):
    """
    Plot performance over time.
    
    Args:
        tracker: PerformanceTracker instance
        metric: Metric to plot
        save_path: Where to save the plot
    """
    if not tracker.history:
        logger.warning("[Viz] No history to plot")
        return
    
    # Extract timestamps and metric values
    timestamps = [i for i in range(len(tracker.history))]
    values = [r['quality'].get(metric, 0) for r in tracker.history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values, 'b-', alpha=0.3, label='All predictions')
    
    # Add moving average
    window = min(10, len(values) // 10)
    if window > 1:
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(values)), moving_avg, 'r-', linewidth=2, label=f'{window}-point moving average')
    
    # Mark best and worst
    best_idx = np.argmax(values)
    worst_idx = np.argmin(values)
    plt.scatter([best_idx], [values[best_idx]], color='green', s=100, zorder=5, label='Best')
    plt.scatter([worst_idx], [values[worst_idx]], color='red', s=100, zorder=5, label='Worst')
    
    plt.xlabel('Prediction Number')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Performance History: {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"[Viz] Saved performance history plot to {save_path}")


def plot_parameter_distributions(tracker, param_name, param_type='meta_params', 
                                 save_path='experiments/param_distribution.png'):
    """
    Plot distribution of parameter values in best vs worst predictions.
    
    Args:
        tracker: PerformanceTracker instance
        param_name: Name of parameter to analyze
        param_type: 'meta_params' or 'keras_params'
        save_path: Where to save the plot
    """
    best = tracker.get_best_predictions(n=20)
    worst = tracker.get_worst_predictions(n=20)
    
    # Extract parameter values
    best_values = [r[param_type].get(param_name) for r in best 
                   if param_name in r.get(param_type, {})]
    worst_values = [r[param_type].get(param_name) for r in worst
                    if param_name in r.get(param_type, {})]
    
    if not best_values or not worst_values:
        logger.warning(f"[Viz] Insufficient data for {param_name}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(best_values, bins=15, alpha=0.7, color='green', label='Best', density=True)
    axes[0].hist(worst_values, bins=15, alpha=0.7, color='red', label='Worst', density=True)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Distribution: {param_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([best_values, worst_values], labels=['Best', 'Worst'])
    axes[1].set_ylabel(param_name)
    axes[1].set_title(f'Box Plot: {param_name}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"[Viz] Saved parameter distribution plot to {save_path}")


def plot_parameter_importance(adaptive_search, save_path='experiments/param_importance.png'):
    """
    Plot parameter importance ranking.
    
    Args:
        adaptive_search: AdaptiveSearchSpace instance
        save_path: Where to save the plot
    """
    importance = adaptive_search.get_parameter_importance_ranking()
    
    if not importance:
        logger.warning("[Viz] No importance data available")
        return
    
    # Take top 15 parameters
    top_n = min(15, len(importance))
    names = [f"{name}\n({ptype.split('_')[0]})" for name, _, ptype in importance[:top_n]]
    scores = [score for _, score, _ in importance[:top_n]]
    colors = ['green' if ptype == 'keras_params' else 'blue' 
              for _, _, ptype in importance[:top_n]]
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, scores, color=colors, alpha=0.7)
    plt.yticks(y_pos, names)
    plt.xlabel('Importance Score')
    plt.title('Parameter Importance Ranking\n(Higher = more impact on prediction quality)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"[Viz] Saved parameter importance plot to {save_path}")


def plot_search_space_evolution(tracker, param_name, param_type='meta_params',
                                save_path='experiments/search_space_evolution.png'):
    """
    Plot how parameter values evolved over time.
    
    Args:
        tracker: PerformanceTracker instance
        param_name: Parameter to track
        param_type: 'meta_params' or 'keras_params'
        save_path: Where to save the plot
    """
    # Extract values and timestamps
    values = []
    matches = []
    timestamps = []
    
    for i, record in enumerate(tracker.history):
        if param_name in record.get(param_type, {}):
            values.append(record[param_type][param_name])
            matches.append(record['quality'].get('matches', 0))
            timestamps.append(i)
    
    if not values:
        logger.warning(f"[Viz] No data for {param_name}")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Parameter values over time
    axes[0].scatter(timestamps, values, c=matches, cmap='RdYlGn', alpha=0.6, s=50)
    axes[0].set_ylabel(param_name)
    axes[0].set_title(f'Parameter Evolution: {param_name}\n(Color = prediction quality)')
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                               norm=plt.Normalize(vmin=min(matches), vmax=max(matches)))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0], label='Matches')
    
    # Matches over time
    axes[1].plot(timestamps, matches, 'b-', alpha=0.3)
    axes[1].scatter(timestamps, matches, c='blue', alpha=0.5, s=30)
    axes[1].set_xlabel('Prediction Number')
    axes[1].set_ylabel('Matches')
    axes[1].set_title('Prediction Quality Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"[Viz] Saved search space evolution plot to {save_path}")


def generate_all_visualizations(tracker, adaptive_search, output_dir='experiments/viz'):
    """
    Generate all performance visualizations.
    
    Args:
        tracker: PerformanceTracker instance
        adaptive_search: AdaptiveSearchSpace instance
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("[Viz] Generating all performance visualizations...")
    
    # Performance history
    plot_performance_history(tracker, metric='matches', 
                             save_path=f'{output_dir}/performance_history.png')
    
    # Parameter importance
    plot_parameter_importance(adaptive_search, 
                             save_path=f'{output_dir}/param_importance.png')
    
    # Get top important parameters and plot their distributions
    importance = adaptive_search.get_parameter_importance_ranking()
    for param_name, score, param_type in importance[:5]:
        safe_name = param_name.replace('/', '_').replace(' ', '_')
        plot_parameter_distributions(tracker, param_name, param_type,
                                     save_path=f'{output_dir}/dist_{safe_name}.png')
        plot_search_space_evolution(tracker, param_name, param_type,
                                    save_path=f'{output_dir}/evolution_{safe_name}.png')
    
    logger.info(f"[Viz] All visualizations saved to {output_dir}")
