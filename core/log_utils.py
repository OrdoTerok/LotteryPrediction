"""
Logging utilities for LotteryPrediction.
"""
import logging
import sys
import io
import json

class SilentLogger:
    """
    Logger that suppresses all output. Use for KerasTuner or any noisy library.
    """
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def exception(self, *args, **kwargs): pass
    def setLevel(self, *args, **kwargs): pass
    def addHandler(self, *args, **kwargs): pass
    def removeHandler(self, *args, **kwargs): pass
    def handlers(self): return []
    def propagate(self, *args, **kwargs): pass
    def getChild(self, *args, **kwargs): return self
    def __getattr__(self, name):
        def no_op(*args, **kwargs): pass
        return no_op

class _DevNull(io.TextIOBase):
    def write(self, *args, **kwargs): pass
    def flush(self): pass

def suppress_console():
    """
    Redirect sys.stdout and sys.stderr to suppress all console output.
    Call at the start of your script or before noisy operations.
    """
    sys.stdout = _DevNull()
    sys.stderr = _DevNull()

def setup_logging(log_filename=None):
    """
    Set up logging to file and console.
    Args:
        log_filename (str, optional): If provided, log to this file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # Remove all handlers first (to avoid duplicate logs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    if log_filename:
        fh = logging.FileHandler(log_filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Do NOT add StreamHandler (no console output)
    return logger

def get_logger():
    return logging.getLogger()

def save_json(data, filename):
    """Save a dictionary or list to a JSON file."""
    import numpy as np
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    with open(filename, 'w') as f:
        json.dump(convert(data), f, indent=2)
