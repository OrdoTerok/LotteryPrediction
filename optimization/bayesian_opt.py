
"""
bayesian_opt.py
Bayesian optimization for meta-parameter search in LotteryPrediction.
Uses Optuna for flexible, efficient search.
Selectable via config.META_OPT_METHOD = 'bayesian'.
Supports robust evaluation with cross-validation (config.CV_FOLDS).
Integrates with main workflow for meta-parameter search.
"""
import optuna # type: ignore
import numpy as np
from util import model_utils
import config

# Define the search space for meta-parameters
def bayesian_optimize(var_names, bounds, final_df, n_trials=10):
	def objective(trial):
		# Suggest values for each meta-parameter
		params = []
		for i, (name, (low, high)) in enumerate(zip(var_names, bounds)):
			if isinstance(low, int) and isinstance(high, int):
				val = trial.suggest_int(name, int(low), int(high))
			else:
				val = trial.suggest_float(name, float(low), float(high))
			params.append(val)
		# Set config
		for i, name in enumerate(var_names):
			setattr(config, name, params[i])
		# Evaluate fitness (lower is better)
		try:
			fitness = model_utils.run_full_workflow(final_df, config)
		except Exception as e:
			import logging
			logger = logging.getLogger(__name__)
			logger.error(f"[BayesianOpt] Fitness error: {e}")
			return float('inf')
		return fitness

	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=n_trials, n_jobs=-1)
	best_params = [study.best_params[name] for name in var_names]
	import logging
	logger = logging.getLogger(__name__)
	logger.info(f"[BayesianOpt] Best params: {study.best_params}")
	return best_params
