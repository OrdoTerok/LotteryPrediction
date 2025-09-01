
# Suppress TensorFlow and Keras warnings and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Modular Imports ---
import config
from util.model_utils import run_pipeline
from util.cache import Cache
from util.log_utils import setup_logging
from util.experiment_tracker import ExperimentTracker
from optimization.meta_search import MetaParameterSearch

def main():
    # Setup logging and experiment tracking
    setup_logging()
    tracker = ExperimentTracker()
    cache = Cache()

    # Orchestrate pipeline
    print("[Pipeline] Starting LotteryPrediction modular pipeline...")
    run_pipeline(config)
    print("[Pipeline] Pipeline complete.")

if __name__ == "__main__":
    main()
