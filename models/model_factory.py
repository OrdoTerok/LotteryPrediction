
"""
models.model_factory
-------------------
Factory for instantiating models by type string. Supports LSTM, RNN, MLP, and LightGBM models.
"""
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
    """
    Retrieve and instantiate a model by type string.
    Args:
        model_type: String identifier for the model type (e.g., 'lstm', 'rnn', 'mlp', 'lgbm').
        *args: Positional arguments to pass to the model constructor.
        **kwargs: Keyword arguments to pass to the model constructor.
    Returns:
        Instantiated model object.
    Raises:
        ValueError: If the model_type is not recognized.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](*args, **kwargs)
