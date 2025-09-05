
# Suppress TensorFlow and Keras warnings and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Modular Imports ---
import sys
import argparse
import config.config as config
from core.cache import Cache
from core.log_utils import setup_logging
from pipeline.experiment_tracker import ExperimentTracker
import cProfile
import datetime
import os


def main():
    parser = argparse.ArgumentParser(description='LotteryPrediction main entry point.')
    parser.add_argument('--cli', choices=['train', 'evaluate', 'tune'], help='Run CLI entry point from scripts/.')
    parser.add_argument('--config', type=str, default='config/config.py', help='Path to config file (for CLI mode)')
    args, unknown = parser.parse_known_args()

    if args.cli:
        script_map = {
            'train': 'scripts/train.py',
            'evaluate': 'scripts/evaluate.py',
            'tune': 'scripts/tune.py',
        }
        script_path = script_map[args.cli]
        # Build command to run the script with any extra args
        cmd = [sys.executable, script_path, '--config', args.config] + unknown
        os.execv(sys.executable, [sys.executable] + cmd[1:])
        return

    # Default: run the main pipeline as before
    # Setup logging and experiment tracking
    log_filename = setup_logging()
    if getattr(config, 'DEVELOPMENT_MODE', False):
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
    # Import run_pipeline from the correct location
    try:
        from pipeline.run_pipeline import run_pipeline
    except ImportError:
        logger.error("Could not import run_pipeline from pipeline.run_pipeline. Please check your project structure.")
        raise
    run_pipeline(config)
    profiler.disable()
    profiler.dump_stats(profile_path)
    logger.info(f"[Pipeline] Profiling complete. Profile saved to {profile_path}")
    logger.info("[Pipeline] Pipeline complete.")

if __name__ == "__main__":
    main()
