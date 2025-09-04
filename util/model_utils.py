# Standard library imports
import os
import json

def get_results_history(cache=None):
    """
    Loads and caches the results_predictions_history.json file in memory or via cache utility.
    Returns the cached history on subsequent calls.
    """
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    if cache is not None:
        history = cache.get(history_path)
        if history is not None:
            return history
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    else:
        history = []
    if cache is not None:
        cache.set(history_path, history)
    return history


