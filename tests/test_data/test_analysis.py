import unittest
import numpy as np
import pandas as pd
from data.analysis import analyze_value_ranges_per_ball

class TestAnalysis(unittest.TestCase):
    def test_analyze_value_ranges_per_ball(self):
        df = pd.DataFrame(np.random.randint(1, 70, size=(10, 6)))
        result = analyze_value_ranges_per_ball(df)
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()
