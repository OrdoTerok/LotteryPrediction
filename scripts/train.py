"""
Train CLI entry point for LotteryPrediction.
"""
import argparse
from main import run_pipeline

def main():
    parser = argparse.ArgumentParser(description='Train LotteryPrediction pipeline.')
    parser.add_argument('--config', type=str, default='config/config.py', help='Path to config file')
    args = parser.parse_args()
    # Dynamically import config
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    run_pipeline(config)

if __name__ == '__main__':
    main()
