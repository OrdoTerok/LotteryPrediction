import unittest
import pandas as pd
from data.split import split_dataframe_by_percentage, split_train_test

class TestSplit(unittest.TestCase):
    def test_split_dataframe_by_percentage(self):
        df = pd.DataFrame({'a': range(10)})
        df1, df2 = split_dataframe_by_percentage(df, 0.7, random_state=42)
        self.assertEqual(len(df1) + len(df2), 10)

    def test_split_train_test(self):
        df = pd.DataFrame({'a': range(10)})
        train, test = split_train_test(df, test_size=0.2, random_state=42)
        self.assertEqual(len(train) + len(test), 10)

if __name__ == '__main__':
    unittest.main()
