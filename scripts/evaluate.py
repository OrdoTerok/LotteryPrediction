"""
Evaluate CLI entry point for LotteryPrediction.
"""
import argparse
from main import run_pipeline

def main():
    parser = argparse.ArgumentParser(description='Evaluate LotteryPrediction pipeline.')
    parser.add_argument('--config', type=str, default='config/config.py', help='Path to config file')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # Set evaluation mode if needed
    if hasattr(config, 'EVAL_MODE'):
        config.EVAL_MODE = True
    else:
        setattr(config, 'EVAL_MODE', True)
    run_pipeline(config)

if __name__ == '__main__':
    main()
