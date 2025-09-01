# models/model_factory.py

from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from models.mlp_model import MLPModel
from models.lgbm_model import LightGBMModel

MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'rnn': RNNModel,
    'mlp': MLPModel,
    'lgbm': LightGBMModel,
}

def get_model(model_type, *args, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](*args, **kwargs)
