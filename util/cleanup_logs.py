import os
import glob

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
MAX_LOGS = 10

def cleanup_logs(log_dir=LOG_DIR, max_logs=MAX_LOGS):
    log_files = sorted(
        glob.glob(os.path.join(log_dir, '*')),
        key=os.path.getmtime,
        reverse=True
    )
    for old_log in log_files[max_logs:]:
        try:
            os.remove(old_log)
            print(f"Deleted old log: {old_log}")
        except Exception as e:
            print(f"Failed to delete {old_log}: {e}")

if __name__ == '__main__':
    cleanup_logs()
