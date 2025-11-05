"""
In-memory and disk caching utilities for LotteryPrediction.
"""
import os
import pickle
import threading
import hashlib

def make_cache_key(*args, **kwargs):
    """
    Generate a stable cache key from arbitrary arguments.
    
    Parameters
    ----------
    *args : tuple
        Positional arguments to include in key.
    **kwargs : dict
        Keyword arguments to include in key.
        
    Returns
    -------
    str
        Hash string suitable for use as cache key.
    """
    # Lazy imports to avoid module-level pandas/numpy dependency
    try:
        import pandas as pd
        import numpy as np
        has_pandas = True
        has_numpy = True
    except ImportError:
        has_pandas = False
        has_numpy = False
    
    key_parts = []
    
    for arg in args:
        if has_pandas and hasattr(pd, 'DataFrame') and isinstance(arg, pd.DataFrame):
            # Use DataFrame shape and first/last row hash
            key_parts.append(f"df_{arg.shape}_{hash(tuple(arg.columns))}")
            if len(arg) > 0:
                key_parts.append(str(hash(tuple(arg.iloc[0].values))))
                key_parts.append(str(hash(tuple(arg.iloc[-1].values))))
        elif has_numpy and hasattr(np, 'ndarray') and isinstance(arg, np.ndarray):
            key_parts.append(f"arr_{arg.shape}_{arg.dtype}")
        elif isinstance(arg, (list, tuple)):
            key_parts.append(f"seq_{len(arg)}_{hash(tuple(str(x) for x in arg))}")
        else:
            key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    # Create hash of combined key parts
    combined = "_".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()

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
    
    def get_stats(self):
        """Get cache statistics."""
        with self.lock:
            mem_count = len(self.memory)
            disk_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            disk_count = len(disk_files)
            return {'memory_entries': mem_count, 'disk_entries': disk_count}


class CVFoldCache:
    """
    Cache for cross-validation fold indices.
    """
    def __init__(self, cache_dir=None, logger=None):
        self.cache = Cache(cache_dir or os.path.join(os.getcwd(), 'cache', 'cv_folds'))
        self.logger = logger
        self.hits = 0
        self.misses = 0
    
    def get_fold_indices(self, n_samples, n_folds, random_state=42):
        """Get cached fold indices."""
        cache_key = f"cv_folds_{n_samples}_{n_folds}_{random_state}"
        result = self.cache.get(cache_key)
        if result is not None:
            self.hits += 1
            if self.logger:
                self.logger.info(f"[CVFoldCache HIT] Using cached folds for n={n_samples}, k={n_folds}")
        else:
            self.misses += 1
        return result
    
    def set_fold_indices(self, n_samples, n_folds, fold_indices, random_state=42):
        """Cache fold indices."""
        cache_key = f"cv_folds_{n_samples}_{n_folds}_{random_state}"
        self.cache.set(cache_key, fold_indices, persist=True)
        if self.logger:
            self.logger.debug(f"[CVFoldCache SET] Cached folds for n={n_samples}, k={n_folds}")
    
    def get_stats(self):
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['hits'] = self.hits
        stats['misses'] = self.misses
        stats['hit_rate'] = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
        return stats


class FeatureCache:
    """
    Cache for feature engineering operations (meta columns, flattening, etc).
    """
    def __init__(self, cache_dir=None, logger=None):
        self.cache = Cache(cache_dir or os.path.join(os.getcwd(), 'cache', 'features'))
        self.logger = logger
        self.hits = 0
        self.misses = 0
    
    def get_meta_cols(self, df):
        """Get cached meta column list for a dataframe."""
        cache_key = make_cache_key(df, 'meta_cols')
        result = self.cache.get(cache_key)
        if result is not None:
            self.hits += 1
            if self.logger:
                self.logger.debug(f"[FeatureCache HIT] meta_cols")
        else:
            self.misses += 1
        return result
    
    def set_meta_cols(self, df, meta_cols):
        """Cache meta column list for a dataframe."""
        cache_key = make_cache_key(df, 'meta_cols')
        self.cache.set(cache_key, meta_cols, persist=False)  # Keep in memory only
    
    def get_flattened_data(self, X, data_id):
        """Get cached flattened data."""
        cache_key = make_cache_key(X, data_id, 'flatten')
        result = self.cache.get(cache_key)
        if result is not None:
            self.hits += 1
            if self.logger:
                self.logger.debug(f"[FeatureCache HIT] flatten_{data_id}")
        else:
            self.misses += 1
        return result
    
    def set_flattened_data(self, X, data_id, X_flat):
        """Cache flattened data."""
        cache_key = make_cache_key(X, data_id, 'flatten')
        self.cache.set(cache_key, X_flat, persist=False)  # Keep in memory only
    
    def get_stats(self):
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['hits'] = self.hits
        stats['misses'] = self.misses
        stats['hit_rate'] = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
        return stats


class PreprocessingCache:
    """
    Specialized cache for preprocessing operations with logging.
    """
    def __init__(self, cache_dir=None, logger=None):
        self.cache = Cache(cache_dir or os.path.join(os.getcwd(), 'cache', 'preprocessing'))
        self.logger = logger
        self.hits = 0
        self.misses = 0
    
    def get_prepared_data(self, df, look_back, meta_cols=None):
        """
        Get cached prepared LSTM data or return None.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        look_back : int
            Window size.
        meta_cols : list, optional
            Meta column names.
            
        Returns
        -------
        tuple or None
            (X, y) if cached, None otherwise.
        """
        cache_key = make_cache_key(df, look_back, meta_cols)
        result = self.cache.get(cache_key)
        
        if result is not None:
            self.hits += 1
            if self.logger:
                self.logger.info(f"[Cache HIT] prepare_data_for_lstm (hits: {self.hits}, misses: {self.misses})")
        else:
            self.misses += 1
            if self.logger:
                self.logger.debug(f"[Cache MISS] prepare_data_for_lstm (hits: {self.hits}, misses: {self.misses})")
        
        return result
    
    def set_prepared_data(self, df, look_back, meta_cols, X, y):
        """
        Cache prepared LSTM data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        look_back : int
            Window size.
        meta_cols : list, optional
            Meta column names.
        X : np.ndarray
            Input sequences.
        y : tuple
            Target arrays.
        """
        cache_key = make_cache_key(df, look_back, meta_cols)
        self.cache.set(cache_key, (X, y), persist=True)
        if self.logger:
            self.logger.debug(f"[Cache SET] prepare_data_for_lstm key={cache_key[:8]}...")
    
    def get_stats(self):
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['hits'] = self.hits
        stats['misses'] = self.misses
        stats['hit_rate'] = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
        return stats
