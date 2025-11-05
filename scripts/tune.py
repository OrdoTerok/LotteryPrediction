"""
Hyperparameter tuning CLI entry point for LotteryPrediction.
"""
import argparse
from pipeline.run_pipeline import run_meta_optimization

def main():
    parser = argparse.ArgumentParser(description='Tune LotteryPrediction pipeline hyperparameters.')
    parser.add_argument('--config', type=str, default='config/config.py', help='Path to config file')
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # Dummy DataFrame for demonstration; replace with actual data loading as needed
    import pandas as pd
    final_df = pd.DataFrame()  # TODO: Replace with actual data
    run_meta_optimization(final_df, config)

if __name__ == '__main__':
    main()
