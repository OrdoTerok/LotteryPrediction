import unittest
import numpy as np
import pandas as pd
from data.augmentation import add_gaussian_noise, pseudo_label

class TestAugmentation(unittest.TestCase):
    def test_add_gaussian_noise(self):
        X = np.zeros((10, 5))
        X_noisy = add_gaussian_noise(X, std=0.5, random_state=42)
        self.assertEqual(X.shape, X_noisy.shape)
        self.assertFalse(np.allclose(X, X_noisy))

    def test_pseudo_label(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        class DummyModel:
            def predict(self, X):
                return np.array([0.95, 0.8, 0.99])
        model = DummyModel()
        result = pseudo_label(df, model, threshold=0.9)
        self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
