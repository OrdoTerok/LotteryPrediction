
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
import config.config as config

# Define the search space for meta-parameters
def bayesian_optimize(var_names, bounds, final_df, n_trials=10, cv=1):
	def objective(trial):
		params = []
		for i, (name, (low, high)) in enumerate(zip(var_names, bounds)):
			if isinstance(low, int) and isinstance(high, int):
				val = trial.suggest_int(name, int(low), int(high))
			else:
				val = trial.suggest_float(name, float(low), float(high))
			params.append(val)
		for i, name in enumerate(var_names):
			setattr(config, name, params[i])
		try:
			# If cv > 1, use cross-validation
			if cv > 1:
				from models.model_factory import get_model
				from data.preprocessing import prepare_data_for_lstm
				look_back_window = getattr(config, 'LOOK_BACK_WINDOW', 10)
				train_df, _ = final_df if isinstance(final_df, tuple) else (final_df, None)
				X_train, y_train = prepare_data_for_lstm(train_df, look_back=look_back_window)
				model_type = getattr(config, 'MODEL_TYPE', 'lstm')
				model = get_model(model_type, input_shape=X_train.shape[1:])
				cv_results = model.cross_validate(X_train, y_train, cv=cv)
				# Use mean of first metric as fitness
				if isinstance(cv_results[0], (list, tuple, np.ndarray)):
					fitness = float(np.mean([r[0] for r in cv_results]))
				else:
					fitness = float(np.mean(cv_results))
			else:
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
