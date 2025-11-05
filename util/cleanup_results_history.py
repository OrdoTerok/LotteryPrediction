import os
import json

HISTORY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data_sets', 'results_predictions_history.json')
MAX_ENTRIES = 10

def cleanup_results_history(history_path=HISTORY_PATH, max_entries=MAX_ENTRIES):
    if not os.path.exists(history_path):
        return
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return
        # Keep only the last max_entries
        new_data = data[-max_entries:]
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
        print(f"Trimmed {history_path} to {len(new_data)} most recent entries.")
    except Exception as e:
        print(f"Failed to trim {history_path}: {e}")

if __name__ == '__main__':
    cleanup_results_history()
