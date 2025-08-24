# particle_swarm.py
"""
Particle Swarm Optimizer for hyperparameter tuning in LotteryPrediction.
Fitness is based on predicted std and KL divergence of model predictions.
"""
import numpy as np
import copy
import config
import importlib
import main

class Particle:
    def __init__(self, bounds, var_names):
        self.position = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.var_names = var_names

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(2)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (global_best - self.position)
        )

    def update_position(self, bounds):
        self.position += self.velocity
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)

    def set_vars(self):
        # Set config variables dynamically
        for i, name in enumerate(self.var_names):
            setattr(config, name, type(getattr(config, name))(self.position[i]))

    def evaluate(self, fitness_func):
        self.set_vars()
        metrics = fitness_func()
        # Fitness: minimize KL, maximize std, maximize accuracy, minimize log loss, maximize entropy
        # You can adjust weights as needed
        fitness = (
            np.mean(metrics['kls'])
            - 0.5 * np.mean(metrics['stds'])
            - 0.5 * np.mean(metrics['accs'])
            + 0.2 * np.mean(metrics['log_losses'])
            - 0.1 * np.mean(metrics['entropies'])
        )
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        return fitness

def fitness_func():
    # Reload main to re-import config
    importlib.reload(config)
    importlib.reload(main)
    # Run main, capture all metrics
    try:
        metrics = main.run_for_pso()
        return metrics
    except Exception as e:
        print("[PSO] Fitness error:", e)
        return {
            'stds': [0],
            'kls': [float('inf')],
            'accs': [0],
            'entropies': [0],
            'log_losses': [float('inf')]
        }

def particle_swarm_optimize(var_names, bounds, n_particles=5, n_iter=10):
    swarm = [Particle(bounds, var_names) for _ in range(n_particles)]
    global_best = None
    global_best_fitness = float('inf')
    for it in range(n_iter):
        print(f"[PSO] Iteration {it+1}/{n_iter}")
        for p in swarm:
            fitness = p.evaluate(fitness_func)
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best = p.position.copy()
        for p in swarm:
            p.update_velocity(global_best)
            p.update_position(bounds)
        print(f"[PSO] Best fitness so far: {global_best_fitness}")
    print("[PSO] Optimization complete. Best position:", global_best)
    return global_best

if __name__ == "__main__":
    # Example: tune LABEL_SMOOTHING and UNIFORM_MIX_PROB
    var_names = ["LABEL_SMOOTHING", "UNIFORM_MIX_PROB"]
    bounds = [(0.0, 0.3), (0.0, 0.3)]
    best = particle_swarm_optimize(var_names, bounds)
    print("Best hyperparameters:", dict(zip(var_names, best)))
