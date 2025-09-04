import unittest
from models.model_factory import get_model

class TestModelFactory(unittest.TestCase):
    def test_get_lstm(self):
        model = get_model('lstm', input_shape=(2, 3))
        self.assertIsNotNone(model)
    def test_get_rnn(self):
        model = get_model('rnn', input_shape=(2, 3))
        self.assertIsNotNone(model)
    def test_get_mlp(self):
        model = get_model('mlp', input_shape=(6,))
        self.assertIsNotNone(model)
    def test_get_lgbm(self):
        model = get_model('lgbm')
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
