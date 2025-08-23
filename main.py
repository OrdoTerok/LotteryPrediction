# This script is a main entry point to download Powerball data.
# It assumes the "Powerball Data Loader" script is saved as a file
# in the same directory, for example, 'powerball_loader.py'.

# You can adjust the import statement based on your file name.
# For this example, we'll assume the file is named 'powerball_loader'.
from data.data_handler import DataHandler
from models.lstm_model import LSTMModel
from typing import Tuple

DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'
KAGGLE_CSV_FILE = 'powerball_usa.csv'

def main():
    """
    Main function to orchestrate the data download process.
    """
    print("Starting the Powerball data download process...")
    
    # Call the function from the imported module to get the DataFrame.
    datagov_df = DataHandler.fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = DataHandler.load_data_from_kaggle(KAGGLE_CSV_FILE)
    
    # Check if the function returned a valid DataFrame.
    if not datagov_df.empty or not kaggle_df.empty:
        print("\nSuccessfully loaded the Powerball data into a DataFrame.")
        print(f"Data includes {kaggle_df.shape[0]} draws and has {kaggle_df.shape[1]} columns.")

        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)

        # Save the DataFrame to a Parquet file.
        DataHandler.save_to_file(final_df)

        # Split the DataFrame into training and testing sets.
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, 0.8)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")

        # The 'look_back' value determines how many previous draws the model
        # will use to predict the next one.
        look_back_window = 10
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

        tuner = kt.RandomSearch(
            lambda hp: LSTMModel.build_lstm_model(hp, input_shape),
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='hypertune_dir',
            project_name='lstm_lottery'
        )

        tuner.search(
            X_train_reshaped, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        print("\nEvaluating the best hypertuned model on the test data...")
        test_loss = best_model.evaluate(X_test_reshaped, y_test, verbose=0)
        print(f"Best Test Loss (Mean Squared Error): {test_loss:.2f}")
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")

if __name__ == "__main__":
    main()
