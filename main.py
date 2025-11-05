import os
import sys

# Suppress TensorFlow and Keras warnings FIRST (before any imports that might load TF)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import argparse
import logging

# Disable default logging to console BEFORE any other imports
logging.lastResort = None
# Remove any existing handlers from root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL + 1)  # Effectively disable until setup_logging
for handler in root_logger.handlers[:]:
    handler.close()
    root_logger.removeHandler(handler)

import config.config as config
from core.cache import Cache
from core.log_utils import setup_logging
from pipeline.experiment_tracker import ExperimentTracker
import cProfile
import datetime
import warnings

# Suppress ALL warnings from being printed to console
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Redirect warnings to logging system instead of stderr
logging.captureWarnings(True)
warnings_logger = logging.getLogger('py.warnings')
warnings_logger.setLevel(logging.ERROR)
warnings_logger.propagate = False

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
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f'log_{timestamp}.rtf')
    log_to_console = getattr(config, 'LOG_TO_CONSOLE', False)
    setup_logging(log_filename, log_to_console=log_to_console)
    
    # If not logging to console, completely suppress stdout/stderr
    if not log_to_console:
        from core.log_utils import suppress_console
        suppress_console()
    
    # Suppress TensorFlow's internal logging
    try:
        import tensorflow as tf
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(logging.ERROR)
        tf_logger.propagate = False
        tf.get_logger().setLevel('ERROR')
    except:
        pass
    
    # Ensure all model loggers propagate to root and are set to INFO
    import logging
    for model_logger_name in [
        'models.lstm_model',
        'models.rnn_model',
        'models.mlp_model',
        'models.lgbm_model',
    ]:
        model_logger = logging.getLogger(model_logger_name)
        model_logger.setLevel(logging.INFO)
        model_logger.propagate = True
    
    # Log development mode to file only, not console
    if getattr(config, 'DEVELOPMENT_MODE', False):
        logger = logging.getLogger(__name__)
        logger.info("[CONFIG] DEVELOPMENT_MODE is ON: Using low values for PSO_PARTICLES, PSO_ITER, and KERAS_TUNER_MAX_TRIALS.")
    
    tracker = ExperimentTracker()
    cache = Cache()

    # Orchestrate pipeline with profiling
    import logging
    logger = logging.getLogger(__name__)
    profs_dir = os.path.join(os.path.dirname(__file__), 'profiles')
    os.makedirs(profs_dir, exist_ok=True)
    logger.info("[Pipeline] Starting LotteryPrediction modular pipeline...")
    profile_path = os.path.join(profs_dir, f'profile_{timestamp}.prof')
    profiler = cProfile.Profile()
    profiler.enable()
    logger.info("[Pipeline] Running pipeline from Main...")
    # Import run_pipeline from the correct location
    try:
        from pipeline.run_pipeline import run_pipeline
    except ImportError:
        logger.error("Could not import run_pipeline from pipeline.run_pipeline. Please check your project structure.")
        raise
    best_pred = None
    best_entry = run_pipeline(config, best_pred=best_pred)
    profiler.disable()
    profiler.dump_stats(profile_path)
    logger.info(f"[Pipeline] Profiling complete.")
    logger.info("[Pipeline] Pipeline complete.")
    # Save the best prediction to results_predictions_history.json
    try:
        from core.log_utils import save_json
        history_path = os.path.join(os.path.dirname(__file__), 'data_sets', 'results_predictions_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = []
        # Validate best_entry before appending
        required_fields = ['timestamp', 'source', 'first_five', 'sixth', 'metrics', 'matches']
        is_complete = all(
            best_entry.get(field) is not None for field in required_fields
        )
        if is_complete:
            history_data.append(best_entry)
            save_json(history_data, history_path)
            logger.info(f"Saved best prediction to {history_path}")
        else:
            logger.warning(f"Skipped saving incomplete best prediction: {best_entry}")
    except Exception as e:
        logger.error(f"Failed to save best prediction to results_predictions_history.json: {e}")
    # Clean up logs: keep only the 10 most recent log files
    try:
        from util.cleanup_logs import cleanup_logs
        cleanup_logs()
        logger.info("[Pipeline] Log cleanup complete.")
    except Exception as e:
        logger.warning(f"[Pipeline] Log cleanup failed: {e}")
    # Clean up results_predictions_history.json: keep only the 10 most recent entries
    try:
        from util.cleanup_results_history import cleanup_results_history
        cleanup_results_history()
        logger.info("[Pipeline] Results history cleanup complete.")
    except Exception as e:
        logger.warning(f"[Pipeline] Results history cleanup failed: {e}")

if __name__ == "__main__":
    main()
