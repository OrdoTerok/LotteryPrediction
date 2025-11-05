"""
Experiment Tracker Module
========================
This module provides the `ExperimentTracker` class for tracking pipeline runs, metrics, and artifacts in the LotteryPrediction project.
It manages experiment directories, meta information, and artifact logging for reproducibility and analysis.
"""
import os
import json
import datetime
import shutil

class ExperimentTracker:
    def __init__(self, base_dir="experiments"):
        """
        Initialize an ExperimentTracker.

        Args:
            base_dir (str): Base directory for experiment runs.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_dir = None
        self.meta = {}

    def start_run(self, config_dict):
        """
        Start a new experiment run and create a run directory.

        Args:
            config_dict (dict): Configuration dictionary for the run.
        """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{now}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.meta["start_time"] = now
        self.meta["config"] = config_dict
        self.save_meta()

    def log_metric(self, name, value):
        """
        Log a metric for the current run.

        Args:
            name (str): Metric name.
            value: Metric value.
        """
        if "metrics" not in self.meta:
            self.meta["metrics"] = {}
        self.meta["metrics"][name] = value
        self.save_meta()

    def log_artifact(self, file_path, artifact_name=None):
        """
        Log an artifact (file) for the current run.

        Args:
            file_path (str): Path to the artifact file.
            artifact_name (str, optional): Name for the artifact in the run directory.
        """
        if not self.run_dir:
            raise RuntimeError("Call start_run() first!")
        if not artifact_name:
            artifact_name = os.path.basename(file_path)
        dest = os.path.join(self.run_dir, artifact_name)
        shutil.copy2(file_path, dest)

    def save_meta(self):
        """
        Save meta information for the current run to meta.json.
        """
        if not self.run_dir:
            return
        meta_path = os.path.join(self.run_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

    def end_run(self):
        """
        End the current experiment run and record end time.
        """
        self.meta["end_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_meta()
