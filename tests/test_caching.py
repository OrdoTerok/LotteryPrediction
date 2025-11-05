"""
Test caching implementation for preprocessing and feature engineering.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cache import Cache, PreprocessingCache, FeatureCache, CVFoldCache, make_cache_key


class TestCacheKey(unittest.TestCase):
    """Test cache key generation."""
    
    def test_make_cache_key_basic(self):
        """Test basic cache key generation."""
        key1 = make_cache_key(1, 2, 3)
        key2 = make_cache_key(1, 2, 3)
        key3 = make_cache_key(1, 2, 4)
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
    
    def test_make_cache_key_with_arrays(self):
        """Test cache key generation with numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])
        
        key1 = make_cache_key(arr1)
        key2 = make_cache_key(arr2)
        key3 = make_cache_key(arr3)
        
        # Same shape and dtype should produce same key
        self.assertEqual(key1, key2)
        # Different values should still produce same key (only shape/dtype matter)
        self.assertEqual(key1, key3)
    
    def test_make_cache_key_with_dataframe(self):
        """Test cache key generation with pandas DataFrame."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df3 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        
        key1 = make_cache_key(df1)
        key2 = make_cache_key(df2)
        key3 = make_cache_key(df3)
        
        # Same shape and columns should produce related keys
        self.assertEqual(key1, key2)
        # Different values should produce different keys (includes first/last row)
        self.assertNotEqual(key1, key3)


class TestPreprocessingCache(unittest.TestCase):
    """Test preprocessing cache functionality."""
    
    def setUp(self):
        """Set up test cache."""
        self.cache = PreprocessingCache()
    
    def test_cache_miss_then_hit(self):
        """Test cache miss followed by hit."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        look_back = 2
        meta_cols = ['col1']
        
        # First call should be a miss
        result = self.cache.get_prepared_data(df, look_back, meta_cols)
        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)
        self.assertEqual(self.cache.hits, 0)
        
        # Set data
        X = np.array([[[1, 2], [3, 4]]])
        y = (np.array([[[1, 0]]]), np.array([[[0, 1]]]))
        self.cache.set_prepared_data(df, look_back, meta_cols, X, y)
        
        # Second call should be a hit
        result = self.cache.get_prepared_data(df, look_back, meta_cols)
        self.assertIsNotNone(result)
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)
        
        # Verify data integrity
        X_cached, y_cached = result
        np.testing.assert_array_equal(X_cached, X)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('hit_rate', stats)


class TestFeatureCache(unittest.TestCase):
    """Test feature cache functionality."""
    
    def setUp(self):
        """Set up test cache."""
        self.cache = FeatureCache()
    
    def test_meta_cols_cache(self):
        """Test meta columns caching."""
        df = pd.DataFrame({'prev_pred_ball_1': [1, 2], 'other': [3, 4]})
        meta_cols = ['prev_pred_ball_1']
        
        # First call should be a miss
        result = self.cache.get_meta_cols(df)
        self.assertIsNone(result)
        
        # Set and retrieve
        self.cache.set_meta_cols(df, meta_cols)
        result = self.cache.get_meta_cols(df)
        self.assertEqual(result, meta_cols)
    
    def test_flattened_data_cache(self):
        """Test flattened data caching."""
        X = np.random.rand(10, 5, 3)
        data_id = 'test_data'
        
        # First call should be a miss
        result = self.cache.get_flattened_data(X, data_id)
        self.assertIsNone(result)
        
        # Set and retrieve
        X_flat = X.reshape(10, -1)
        self.cache.set_flattened_data(X, data_id, X_flat)
        result = self.cache.get_flattened_data(X, data_id)
        np.testing.assert_array_equal(result, X_flat)


class TestCVFoldCache(unittest.TestCase):
    """Test CV fold cache functionality."""
    
    def setUp(self):
        """Set up test cache."""
        self.cache = CVFoldCache()
    
    def test_fold_indices_cache(self):
        """Test fold indices caching."""
        n_samples = 100
        n_folds = 5
        random_state = 42
        
        # First call should be a miss
        result = self.cache.get_fold_indices(n_samples, n_folds, random_state)
        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)
        
        # Set fold indices
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_indices = list(kf.split(np.arange(n_samples)))
        self.cache.set_fold_indices(n_samples, n_folds, fold_indices, random_state)
        
        # Second call should be a hit
        result = self.cache.get_fold_indices(n_samples, n_folds, random_state)
        self.assertIsNotNone(result)
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(len(result), n_folds)
        
        # Verify fold indices are the same
        for (train1, val1), (train2, val2) in zip(fold_indices, result):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)
    
    def test_different_params_different_cache(self):
        """Test that different parameters produce different cache entries."""
        n_samples = 100
        n_folds = 5
        
        from sklearn.model_selection import KFold
        
        # Set folds with random_state=42
        kf1 = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_indices_1 = list(kf1.split(np.arange(n_samples)))
        self.cache.set_fold_indices(n_samples, n_folds, fold_indices_1, random_state=42)
        
        # Set folds with random_state=43
        kf2 = KFold(n_splits=n_folds, shuffle=True, random_state=43)
        fold_indices_2 = list(kf2.split(np.arange(n_samples)))
        self.cache.set_fold_indices(n_samples, n_folds, fold_indices_2, random_state=43)
        
        # Retrieve both
        result_1 = self.cache.get_fold_indices(n_samples, n_folds, random_state=42)
        result_2 = self.cache.get_fold_indices(n_samples, n_folds, random_state=43)
        
        # They should be different
        train_1, _ = result_1[0]
        train_2, _ = result_2[0]
        self.assertFalse(np.array_equal(train_1, train_2))


if __name__ == '__main__':
    unittest.main()
