"""
Unified meta-parameter search interface for LotteryPrediction.

This module provides the MetaParameterSearch class, which wraps both
Particle Swarm Optimization (PSO) and Bayesian optimization for hyperparameter search.
It delegates to the appropriate optimizer based on the selected method.

Typical usage:
    from optimization.meta_search import MetaParameterSearch
    searcher = MetaParameterSearch(method='pso' or 'bayesian')
    best_params = searcher.search(var_names, bounds, final_df, ...)

All meta-search orchestration logic is contained here.
"""
from optimization.particle_swarm import particle_swarm_optimize
from optimization.bayesian_opt import bayesian_optimize

class MetaParameterSearch:
    def __init__(self, method='pso'):
        """
        Initialize the MetaParameterSearch interface.
        Args:
            method: Optimization method to use ('pso' or 'bayesian').
        """
        self.method = method.lower()

    def search(self, var_names, bounds, final_df, n_trials=10, n_particles=5, n_iter=10):
        """
        Run meta-parameter search using the selected optimization method.
        Args:
            var_names: List of parameter names.
            bounds: List of (low, high) tuples for each parameter.
            final_df: Tuple of (train_df, test_df) DataFrames or a single DataFrame.
            n_trials: Number of trials for Bayesian optimization.
            n_particles: Number of particles for PSO.
            n_iter: Number of iterations for PSO.
        Returns:
            List of best parameter values found by the selected optimizer.
        Raises:
            ValueError: If the optimization method is not recognized.
        """
        if self.method == 'bayesian':
            return bayesian_optimize(var_names, bounds, final_df, n_trials=n_trials)
        elif self.method == 'pso':
            return particle_swarm_optimize(var_names, bounds, final_df, n_particles=n_particles, n_iter=n_iter)
        else:
            raise ValueError(f"Unknown meta-parameter search method: {self.method}")
