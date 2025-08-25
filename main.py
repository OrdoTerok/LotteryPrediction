# --- Ensembling support: average predictions from multiple models ---
# This script is a main entry point to download Powerball data.
# It assumes the "Powerball Data Loader" script is saved as a file
# in the same directory, for example, 'powerball_loader.py'.
# You can adjust the import statement based on your file name.
# For this example, we'll assume the file is named 'powerball_loader'.

from data.data_handler import DataHandler
import numpy as np
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel

from util.data_utils import analyze_value_ranges_per_ball
from util.model_utils import run_full_workflow
import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show info and above
print("I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.")
DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'

def main():
    # Diagnostics moved inside data-loaded block
    """
    Main function to orchestrate the data download process.
    """
    print("Starting the Powerball data download process...")
    # Call the function from the imported module to get the DataFrame.
    datagov_df = DataHandler.fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = DataHandler.load_data_from_kaggle(config.KAGGLE_CSV_FILE)
    # Check if the function returned a valid DataFrame.
    if not datagov_df.empty or not kaggle_df.empty:
        print("\nSuccessfully loaded the Powerball data into a DataFrame.")
        print(f"Data includes {kaggle_df.shape[0]} draws and has {kaggle_df.shape[1]} columns.")
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        # Save the DataFrame to a Parquet file.
        DataHandler.save_to_file(final_df)

        # --- ANALYSIS: Most Likely Value Ranges Per Ball ---
        print("\n--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
        analyze_value_ranges_per_ball(final_df)

        # Split the DataFrame into training and testing sets.
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
        # The 'look_back' value determines how many previous draws the model
        # will use to predict the next one.
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        print(f"Prepared training data shape: {X_train.shape}")
        print(f"Prepared testing data shape: {X_test.shape}")
        if X_train.size == 0 or X_test.size == 0:
            print("\nNot enough data to create sequences. Exiting.")
            return
        # Hypertuning with KerasTuner
        import keras_tuner as kt
        print("\nStarting KerasTuner hypertuning...")
        input_shape = (X_train.shape[1] // 6, 6)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
        # --- KerasTuner for model hyperparameters only ---
    if not datagov_df.empty or not kaggle_df.empty:
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        # Save the DataFrame to a Parquet file.
        DataHandler.save_to_file(final_df)

        # --- ANALYSIS: Most Likely Value Ranges Per Ball ---
        print("\n--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
        analyze_value_ranges_per_ball(final_df)

        # --- PSO meta-optimization for meta-hyperparameters only ---
        from particle_swarm import particle_swarm_optimize
        var_names = [
            "LABEL_SMOOTHING",
            "UNIFORM_MIX_PROB",
            "TEMP_MIN",
            "TEMP_MAX",
            "EARLY_STOPPING_PATIENCE"
        ]
        bounds = [
            (0.0, 0.3),      # LABEL_SMOOTHING
            (0.0, 0.3),      # UNIFORM_MIX_PROB
            (0.5, 1.5),      # TEMP_MIN
            (1.5, 2.5),      # TEMP_MAX
            (1, 10)          # EARLY_STOPPING_PATIENCE
        ]
        best = particle_swarm_optimize(var_names, bounds, final_df, n_particles=config.PSO_PARTICLES, n_iter=config.PSO_ITER)
        print("Best meta-hyperparameters:", dict(zip(var_names, best)))

        # Set config to best found by PSO before running KerasTuner/model search
        for i, name in enumerate(var_names):
            setattr(config, name, best[i])

        # Now run KerasTuner for model/training hyperparameters only, using the best meta-parameters
        run_full_workflow(final_df, config)
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")
        
if __name__ == "__main__":
    main()
