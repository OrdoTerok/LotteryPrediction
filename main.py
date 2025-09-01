# Suppress TensorFlow and Keras warnings and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logs (0 = all, 1 = info, 2 = warning, 3 = error)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning)
import config
from util.model_utils import run_pipeline

if __name__ == "__main__":
    run_pipeline(config)
