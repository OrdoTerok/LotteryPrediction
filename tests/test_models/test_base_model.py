import unittest
from models.base_model import BaseModel
import numpy as np

class DummyModel(BaseModel):
    pass

class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_cross_validate_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model.cross_validate(np.zeros((1,1)), np.zeros((1,1)))

    def test_fit_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model.fit(np.zeros((1,1)), np.zeros((1,1)))

    def test_predict_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model.predict(np.zeros((1,1)))

    def test_evaluate_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model.evaluate(np.zeros((1,1)), np.zeros((1,1)))

if __name__ == '__main__':
    unittest.main()
