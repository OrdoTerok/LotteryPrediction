"""
In-memory and disk caching utilities for LotteryPrediction.
"""
import os
import pickle
import threading

class Cache:
    def __init__(self, cache_dir=None):
        self.memory = {}
        self.lock = threading.Lock()
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get(self, key):
        with self.lock:
            if key in self.memory:
                return self.memory[key]
            disk_path = os.path.join(self.cache_dir, f'{key}.pkl')
            if os.path.exists(disk_path):
                with open(disk_path, 'rb') as f:
                    value = pickle.load(f)
                    self.memory[key] = value
                    return value
            return None

    def set(self, key, value, persist=True):
        with self.lock:
            self.memory[key] = value
            if persist:
                disk_path = os.path.join(self.cache_dir, f'{key}.pkl')
                with open(disk_path, 'wb') as f:
                    pickle.dump(value, f)

    def clear(self):
        with self.lock:
            self.memory.clear()
            for fname in os.listdir(self.cache_dir):
                if fname.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, fname))
