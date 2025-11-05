import unittest
import numpy as np
import pandas as pd
from data.preprocessing import combine_and_clean_data, prepare_data_for_lstm, clean_data

class TestPreprocessing(unittest.TestCase):
    def test_combine_and_clean_data(self):
        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'a': [3, 4]})
        df = combine_and_clean_data(df1, df2)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data_for_lstm(self):
        df = pd.DataFrame(np.random.rand(10, 6))
        X, y = prepare_data_for_lstm(df, look_back=2)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, tuple)

    def test_clean_data(self):
        df = pd.DataFrame({'a': [1, None, 3]})
        cleaned = clean_data(df)
        self.assertIsInstance(cleaned, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
