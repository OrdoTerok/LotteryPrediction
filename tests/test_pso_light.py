"""
Test script for PSO with Post-CV/Ensemble approach.
Run this to verify the lighter alternative works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config.config as config
from data.loaders import load_data_from_kaggle
from data.split import split_dataframe_by_percentage
from pipeline.run_pipeline import run_meta_optimization

def test_pso_post_cv_ensemble():
    """
    Test the PSO + Post-CV/Ensemble workflow.
    """
    print("="*60)
    print("Testing PSO with Post-CV/Ensemble (Lighter Alternative)")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading data...")
    kaggle_path = config.KAGGLE_CSV_FILE
    df = load_data_from_kaggle(kaggle_path)
    print(f"Loaded {len(df)} rows")
    
    # Enable lighter approach
    print("\n[2/4] Configuring lighter approach...")
    config.USE_PSO_POST_CV_ENSEMBLE = True
    config.META_OPT_METHOD = 'pso'
    config.PSO_PARTICLES = 2  # Small for testing
    config.PSO_ITER = 2
    config.CV_FOLDS = 3  # Small for testing
    config.DEVELOPMENT_MODE = True
    print(f"  PSO_PARTICLES: {config.PSO_PARTICLES}")
    print(f"  PSO_ITER: {config.PSO_ITER}")
    print(f"  CV_FOLDS: {config.CV_FOLDS}")
    print(f"  USE_PSO_POST_CV_ENSEMBLE: {config.USE_PSO_POST_CV_ENSEMBLE}")
    
    # Run meta-optimization
    print("\n[3/4] Running PSO with Post-CV/Ensemble...")
    print("  This will:")
    print("    a) Run PSO with single train/val split")
    print("    b) Run 3-fold CV on best params")
    print("    c) Run ensemble on best params")
    print("\n  (This may take 5-10 minutes...)\n")
    
    try:
        run_meta_optimization(df, config)
        print("\n[4/4] ✓ Test passed! Lighter approach works correctly.")
    except Exception as e:
        print(f"\n[4/4] ✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("  - PSO search: COMPLETED")
    print("  - Post-PSO CV: COMPLETED")
    print("  - Post-PSO Ensemble: COMPLETED")
    print("  - Overall: SUCCESS")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_pso_post_cv_ensemble()
    sys.exit(0 if success else 1)
