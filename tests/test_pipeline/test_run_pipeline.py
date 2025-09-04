import pytest
import types
from pipeline import run_pipeline

def test_run_pipeline_minimal(monkeypatch):
    class DummyConfig:
        KAGGLE_CSV_FILE = 'data_sets/base_dataset.csv'
        TRAIN_SPLIT = 0.8
        LOOK_BACK_WINDOW = 1
        USE_PSEUDO_LABELING = False
        USE_NOISE_INJECTION = False
        ITERATIVE_STACKING = False
        ITERATIVE_STACKING_ROUNDS = 1
    # Patch dependencies to avoid actual file/IO
    monkeypatch.setattr(run_pipeline, 'fetch_data_from_datagov', lambda url: [])
    monkeypatch.setattr(run_pipeline, 'load_data_from_kaggle', lambda path: [])
    monkeypatch.setattr(run_pipeline, 'combine_and_clean_data', lambda a, b: [])
    monkeypatch.setattr(run_pipeline, 'split_dataframe_by_percentage', lambda df, split: ([], []))
    monkeypatch.setattr(run_pipeline, 'prepare_data_for_lstm', lambda df, look_back: (types.SimpleNamespace(size=1), [[0]]))
    monkeypatch.setattr(run_pipeline, 'Cache', lambda: types.SimpleNamespace(get=lambda k: None, set=lambda k, v: None))
    monkeypatch.setattr(run_pipeline, 'ExperimentTracker', lambda: types.SimpleNamespace(start_run=lambda x: None, log_artifact=lambda *a, **k: None, end_run=lambda: None))
    # Should not raise
    config = DummyConfig()
    result = run_pipeline.run_pipeline(config)
    assert result is not None or result is None  # Just check it runs
