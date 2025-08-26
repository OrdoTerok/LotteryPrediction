import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class NNMetaLearner:
    def __init__(self, input_dim, output_dim, hidden_units=32, lr=0.01, epochs=10, batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(self.hidden_units, activation='relu'),
            layers.Dense(self.output_dim, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(self.lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)
