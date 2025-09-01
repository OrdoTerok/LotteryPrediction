import logging
import datetime

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
    logging.basicConfig(level=logging.INFO, handlers=[file_handler], force=True)
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
