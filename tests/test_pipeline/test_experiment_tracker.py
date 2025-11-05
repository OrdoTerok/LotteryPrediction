import os
import tempfile
import json
from pipeline.experiment_tracker import ExperimentTracker

def test_experiment_tracker_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(base_dir=tmpdir)
        config = {'param': 1}
        tracker.start_run(config)
        tracker.log_metric('accuracy', 0.99)
        # Create a dummy artifact
        artifact_path = os.path.join(tmpdir, 'dummy.txt')
        with open(artifact_path, 'w') as f:
            f.write('test')
        tracker.log_artifact(artifact_path)
        tracker.end_run()
        # Check meta.json exists
        run_dirs = [d for d in os.listdir(tmpdir) if d.startswith('run_')]
        assert run_dirs, 'No run directory created'
        meta_path = os.path.join(tmpdir, run_dirs[0], 'meta.json')
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert 'start_time' in meta and 'end_time' in meta
        assert 'metrics' in meta and meta['metrics']['accuracy'] == 0.99
