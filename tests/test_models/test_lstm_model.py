import unittest
import numpy as np
from models.lstm_model import LSTMModel

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.model = LSTMModel(input_shape=(2, 3))
        self.X = np.random.rand(10, 2, 3)
        self.y = {'first_five': np.random.rand(10, 5, 69), 'sixth': np.random.rand(10, 1, 26)}

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
