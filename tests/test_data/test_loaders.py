import unittest
import pandas as pd
from data.loaders import load_csv

class TestLoaders(unittest.TestCase):
    def test_load_csv(self):
        df = load_csv('data_sets/base_dataset.csv')
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
