import unittest
import numpy as np
from models.meta_learner import NNMetaLearner

class TestNNMetaLearner(unittest.TestCase):
    def setUp(self):
        self.model = NNMetaLearner(input_dim=2, output_dim=3, hidden_units=4, epochs=1)
        self.X = np.random.rand(10, 2)
        self.y = np.random.randint(0, 3, 10)

    def test_fit_predict(self):
        self.model.fit(self.X, self.y)
        preds = self.model.predict_proba(self.X)
        self.assertEqual(preds.shape, (10, 3))

if __name__ == '__main__':
    unittest.main()
