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
