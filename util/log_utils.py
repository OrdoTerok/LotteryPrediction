"""
DEPRECATED: Moved to core/log_utils.py
"""

from .log_utils import suppress_console
...

from .log_utils import suppress_console
suppress_console()
import logging
import datetime

class SilentLogger:
    """
    Logger that suppresses all output. Use for KerasTuner or any noisy library.
    """
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass
    def critical(self, *args, **kwargs):
        pass
    def log(self, *args, **kwargs):
        pass
    def exception(self, *args, **kwargs):
        pass
    def setLevel(self, *args, **kwargs):
        pass
    def addHandler(self, *args, **kwargs):
        pass
    def removeHandler(self, *args, **kwargs):
        pass
    def handlers(self):
        return []
    def propagate(self, *args, **kwargs):
        pass
    def getChild(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        # Suppress any other logger method
        def no_op(*args, **kwargs):
            pass
        return no_op

import sys
import io

class _DevNull(io.TextIOBase):
    def write(self, *args, **kwargs):
        pass
    def flush(self):
        pass

def suppress_console():
    """
    Redirect sys.stdout and sys.stderr to suppress all console output.
    Call at the start of your script or before noisy operations.
    """
    sys.stdout = _DevNull()
    sys.stderr = _DevNull()

def setup_logging(log_filename=None):
    """
    Set up logging to a unique log file per execution, named log_TIMESTAMP.rtf.
    """
    import os
    if log_filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f'log_{timestamp}.rtf')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Add only file handler
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    # Add NullHandler to suppress console output
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    root_logger.addHandler(NullHandler())
    return log_filename

def get_logger(name=None):
    return logging.getLogger(name)
"""
Logging utilities for saving PSO, KerasTuner, and evaluation results.
"""
import json
import csv

def save_json(data, filename):
    """Save a dictionary or list to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_csv(rows, filename, header=None):
    """Save a list of rows (list of lists or dicts) to a CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
