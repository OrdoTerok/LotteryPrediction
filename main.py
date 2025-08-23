# This script is a main entry point to download Powerball data.
# It assumes the "Powerball Data Loader" script is saved as a file
# in the same directory, for example, 'powerball_loader.py'.

# You can adjust the import statement based on your file name.
# For this example, we'll assume the file is named 'powerball_loader'.
import data_handler
import lstm_model
from typing import Tuple

def main():
    """
    Main function to orchestrate the data download process.
    """
    print("Starting the Powerball data download process...")
    
    # Call the function from the imported module to get the DataFrame.
    powerball_df = data_handler.download_powerball_data_from_kaggle()
    
    # Check if the function returned a valid DataFrame.
    if powerball_df is not None:
        print("\nSuccessfully loaded the Powerball data into a DataFrame.")
        print(f"Data includes {powerball_df.shape[0]} draws and has {powerball_df.shape[1]} columns.")

        # Save the DataFrame to a Parquet file.
        data_handler.save_to_parquet(powerball_df)

        # Split the DataFrame into training and testing sets.
        train_df, test_df = data_handler.split_dataframe_by_percentage(powerball_df, 0.8)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")

        # The 'look_back' value determines how many previous draws the model
        # will use to predict the next one.
        look_back_window = 10
        X_train, y_train = data_handler.prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_test, y_test = data_handler.prepare_data_for_lstm(test_df, look_back=look_back_window)
    
        print(f"Prepared training data shape: {X_train.shape}")
        print(f"Prepared testing data shape: {X_test.shape}")
        if X_train.size == 0 or X_test.size == 0:
            print("\nNot enough data to create sequences. Exiting.")
            return
        
        model = lstm_model.build_and_train_lstm(X_train, y_train)

        # Evaluate the model on the test set.
        print("\nEvaluating the model on the test data...")
        test_loss = model.evaluate(X_test.reshape(X_test.shape[0], -1, 5), y_test, verbose=0)
        print(f"Test Loss (Mean Squared Error): {test_loss:.2f}")
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")

if __name__ == "__main__":
    main()
