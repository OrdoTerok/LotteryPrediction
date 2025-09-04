
# Suppress TensorFlow and Keras warnings and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Modular Imports ---

import config.config as config
from util.model_utils import run_pipeline
from util.cache import Cache
from util.log_utils import setup_logging
from util.experiment_tracker import ExperimentTracker
from optimization.meta_search import MetaParameterSearch
import cProfile
import pstats
import datetime
import os

def main():
    # Setup logging and experiment tracking
    log_filename = setup_logging()
    import config.config as config
    if getattr(config, 'DEVELOPMENT_MODE', False):
        import warnings
        warnings.warn("[CONFIG] DEVELOPMENT_MODE is ON: Using low values for PSO_PARTICLES, PSO_ITER, and KERAS_TUNER_MAX_TRIALS.")
    tracker = ExperimentTracker()
    cache = Cache()

    # Orchestrate pipeline with profiling
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[Pipeline] Starting LotteryPrediction modular pipeline...")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    profile_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(profile_dir, exist_ok=True)
    profile_path = os.path.join(profile_dir, f'profile_{timestamp}.prof')
    profiler = cProfile.Profile()
    profiler.enable()
    logger.info("[Pipeline] Running pipeline from Main...")
    run_pipeline(config)
    profiler.disable()
    profiler.dump_stats(profile_path)
    logger.info(f"[Pipeline] Profiling complete. Profile saved to {profile_path}")
    logger.info("[Pipeline] Pipeline complete.")

if __name__ == "__main__":
    main()
