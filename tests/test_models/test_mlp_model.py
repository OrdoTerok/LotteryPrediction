import unittest
import numpy as np
from models.mlp_model import MLPModel

class TestMLPModel(unittest.TestCase):
    def setUp(self):
        self.model = MLPModel(input_shape=(6,), num_first=2, num_first_classes=3, num_sixth_classes=2)
        self.X = np.random.rand(10, 6)
        self.y = {'first_five': np.random.rand(10, 2, 3), 'sixth': np.random.rand(10, 1, 2)}

    def test_fit_predict_evaluate(self):
        self.model.fit(self.X, self.y, epochs=1, batch_size=2)
        preds = self.model.predict(self.X)
        self.assertIsNotNone(preds)
        self.model.evaluate(self.X, self.y)

    def test_cross_validate(self):
        results = self.model.cross_validate(self.X, self.y, cv=2, epochs=1, batch_size=2)
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()
