import unittest
import numpy as np
from models.lgbm_model import LightGBMModel

class TestLightGBMModel(unittest.TestCase):
    def setUp(self):
        self.model = LightGBMModel(num_first=2, num_first_classes=3, num_sixth_classes=2)
        self.X = np.random.rand(10, 4)
        self.y = (np.random.randint(0, 3, (10, 2, 3)), np.random.randint(0, 2, (10, 1, 2)))

    def test_fit_predict_evaluate(self):
        self.model.fit(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertIsInstance(preds, tuple)
        self.model.evaluate(self.X, self.y)

    def test_cross_validate(self):
        results = self.model.cross_validate(self.X, self.y, cv=2)
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()
