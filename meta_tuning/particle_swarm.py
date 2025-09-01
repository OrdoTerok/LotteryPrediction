def check_constraints(var_names, position):
	"""
	Returns True if all constraints are satisfied, False otherwise.
	Add custom constraints here as needed.
	Example: enforce integer, positivity, or relational constraints.
	"""
	# Example: all parameters must be positive
	# for v in position:
	#     if v <= 0:
	#         return False
	# Add more custom constraints as needed
	return True  # No constraints by default
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


def get_valid_initial_value(var_name, low, high):
	"""
	Return a valid, meaningful initial value for a parameter.
	Customize this function for each parameter as needed.
	"""
	# Example: for learning rates, avoid 0 or 1; for units, prefer powers of 2, etc.
	# You can add custom logic per var_name here.
	if 'lr' in var_name.lower() or 'learning_rate' in var_name.lower():
		# For learning rates, use log-uniform sampling in [low, high]
		return float(np.exp(np.random.uniform(np.log(max(low, 1e-5)), np.log(high))))
	elif 'units' in var_name.lower() or 'num_leaves' in var_name.lower():
		# For units, pick a power of 2 within bounds
		min_pow = int(np.ceil(np.log2(max(low, 1))))
		max_pow = int(np.floor(np.log2(high)))
		if min_pow > max_pow:
			return int(2 ** min_pow)
		return int(2 ** np.random.randint(min_pow, max_pow + 1))
	# Default: uniform
	return float(np.random.uniform(low, high))

class Particle:
	def __init__(self, bounds, var_names):
		self.position = np.array([
			get_valid_initial_value(var_name, low, high)
			for (var_name, (low, high)) in zip(var_names, bounds)
		])
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
		# Constraint handling: check before evaluating fitness
		if not check_constraints(self.var_names, self.position):
			print(f"[PSO][DEBUG] Constraint violation for position: {self.position}")
			fitness = 1e6  # Large penalty for constraint violation
		else:
			self.set_vars()
			print(f"[PSO][DEBUG] Evaluating fitness for position: {self.position}")
			fitness = fitness_func()  # Now returns a float (best val_loss)
			if fitness is None or np.isnan(fitness) or np.isinf(fitness):
				print(f"[PSO][DEBUG] Invalid fitness value after evaluation: {fitness}")
				fitness = 1e6  # Large penalty for invalid fitness
		if fitness < self.best_fitness:
			self.best_fitness = fitness
			self.best_position = self.position.copy()
		return fitness

def is_data_valid(train_df, test_df):
	# Check for empty data
	if train_df is None or test_df is None:
		print("[PSO][DEBUG] DataFrame is None.")
		return False
	if hasattr(train_df, 'empty') and train_df.empty:
		print("[PSO][DEBUG] train_df is empty.")
		return False
	if hasattr(test_df, 'empty') and test_df.empty:
		print("[PSO][DEBUG] test_df is empty.")
		return False
	# Check for NaN values
	if hasattr(train_df, 'isnull') and train_df.isnull().values.any():
		print("[PSO][DEBUG] train_df contains NaN values.")
		return False
	if hasattr(test_df, 'isnull') and test_df.isnull().values.any():
		print("[PSO][DEBUG] test_df contains NaN values.")
		return False
	return True

def fitness_func(train_df, test_df):
	# Data validity check
	if not is_data_valid(train_df, test_df):
		print("[PSO] Data invalid: empty or contains NaN.")
		return 1e6  # Large penalty for invalid data
	# Reload config to ensure PSO changes are picked up
	importlib.reload(config)
	from util import model_utils
	# Log current config values for debugging
	print(f"[PSO][DEBUG] Current config values: {[getattr(config, k, None) for k in dir(config) if not k.startswith('__')]}")
	try:
		fitness = model_utils.run_full_workflow(train_df, test_df, config)
		if fitness is None or np.isnan(fitness) or np.isinf(fitness):
			import traceback
			print(f"[PSO][DEBUG] Invalid fitness value returned: {fitness}")
			print("[PSO][DEBUG] Traceback for invalid fitness:")
			traceback.print_stack()
			print(f"[PSO][DEBUG] Config values at invalid fitness: {[getattr(config, k, None) for k in dir(config) if not k.startswith('__')]}")
			return 1e6  # Large penalty for invalid fitness
		return fitness
	except Exception as e:
		import traceback
		print("[PSO] Fitness error:", e)
		traceback.print_exc()
		print(f"[PSO][DEBUG] Config values at error: {[getattr(config, k, None) for k in dir(config) if not k.startswith('__')]}")
		return 1e6  # Large penalty for error

def particle_swarm_optimize(var_names, bounds, final_df, n_particles=5, n_iter=10):
	# Full PSO implementation
	# Expect final_df to be a tuple: (train_df, test_df)
	train_df, test_df = final_df
	particles = [Particle(bounds, var_names) for _ in range(n_particles)]
	global_best_position = None
	global_best_fitness = float('inf')

	from joblib import Parallel, delayed
	for iter in range(n_iter):
		print(f"[PSO] Iteration {iter+1}/{n_iter}")
		# Parallel fitness evaluation
		fitnesses = Parallel(n_jobs=-1)(delayed(lambda p: p.evaluate(lambda: fitness_func(train_df, test_df)))(particle) for particle in particles)
		for idx, (particle, fitness) in enumerate(zip(particles, fitnesses)):
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
