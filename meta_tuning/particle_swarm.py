# particle_swarm.py
"""
Particle Swarm Optimizer for meta-parameter tuning in LotteryPrediction.
Selectable via config.META_OPT_METHOD = 'pso'.
Supports robust evaluation with cross-validation (config.CV_FOLDS).
Integrates with main workflow for meta-parameter search.
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
		fitness = fitness_func()  # Now returns a float (best val_loss)
		if fitness < self.best_fitness:
			self.best_fitness = fitness
			self.best_position = self.position.copy()
		return fitness

def fitness_func(final_df):
	# Reload config to ensure PSO changes are picked up
	importlib.reload(config)
	from util import model_utils
	try:
		fitness = model_utils.run_full_workflow(final_df, config)
		return fitness
	except Exception as e:
		print("[PSO] Fitness error:", e)
		return float('inf')

def particle_swarm_optimize(var_names, bounds, final_df, n_particles=5, n_iter=10):
