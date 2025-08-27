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
import os

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

def fitness_func(train_df, test_df):
	# Reload config to ensure PSO changes are picked up
	importlib.reload(config)
	from util import model_utils
	try:
		fitness = model_utils.run_full_workflow(train_df, test_df, config)
		return fitness
	except Exception as e:
		print("[PSO] Fitness error:", e)
		return float('inf')

def particle_swarm_optimize(var_names, bounds, final_df, n_particles=5, n_iter=10):
	# Full PSO implementation
	# Expect final_df to be a tuple: (train_df, test_df)
	train_df, test_df = final_df
	particles = [Particle(bounds, var_names) for _ in range(n_particles)]
	global_best_position = None
	global_best_fitness = float('inf')

	for iter in range(n_iter):
		print(f"[PSO] Iteration {iter+1}/{n_iter}")
		for idx, particle in enumerate(particles):
			fitness = particle.evaluate(lambda: fitness_func(train_df, test_df))
			print(f"[PSO] Particle {idx+1} fitness: {fitness:.6f}")
			if fitness < global_best_fitness:
				global_best_fitness = fitness
				global_best_position = particle.position.copy()
		# Update velocities and positions
		for particle in particles:
			# If global_best_position is None (first iteration), use particle.best_position
			if global_best_position is None:
				ref_position = particle.best_position
			else:
				ref_position = global_best_position
			particle.update_velocity(ref_position)
			particle.update_position(bounds)
	print(f"[PSO] Best fitness: {global_best_fitness:.6f}")
	print(f"[PSO] Best position: {global_best_position}")
	if global_best_position is None:
		# Fallback: use the best position from the first particle
		print("[PSO] Warning: No global best found, using first particle's best position.")
		return list(particles[0].best_position)
	return list(global_best_position)
