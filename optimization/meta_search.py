"""
Unified meta-parameter search interface for LotteryPrediction.
Provides MetaParameterSearch class to wrap PSO and Bayesian optimization.
"""
from optimization.particle_swarm import particle_swarm_optimize
from optimization.bayesian_opt import bayesian_optimize

class MetaParameterSearch:
    def __init__(self, method='pso'):
        self.method = method.lower()

    def search(self, var_names, bounds, final_df, n_trials=10, n_particles=5, n_iter=10):
        if self.method == 'bayesian':
            return bayesian_optimize(var_names, bounds, final_df, n_trials=n_trials)
        elif self.method == 'pso':
            return particle_swarm_optimize(var_names, bounds, final_df, n_particles=n_particles, n_iter=n_iter)
        else:
            raise ValueError(f"Unknown meta-parameter search method: {self.method}")
