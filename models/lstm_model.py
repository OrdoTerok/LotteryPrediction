import tensorflow as tf
import numpy as np

def build_and_train_lstm(X_train: np.ndarray, y_train: np.ndarray):
    """
    Builds, compiles, and trains a simple LSTM model.

    Args:
        X_train (np.ndarray): The features for training.
        y_train (np.ndarray): The labels/targets for training.
    """
    # The LSTM layer expects input in the shape (samples, time steps, features).
    # In this case, each 'time step' is a winning number set from the 'look_back' window.
    # We reshape the input to be (samples, look_back, num_features_per_timestep).
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 5) # 5 numbers per draw

    # Define the model architecture.
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        tf.keras.layers.Dense(5) # Output layer with 5 units for the 5 winning numbers.
    ])

    # Compile the model.
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model.
    print("\nStarting model training...")
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=1)
    print("Model training complete.")
    
    return model