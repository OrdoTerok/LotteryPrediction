# This script is a main entry point to download Powerball data.
# It assumes the "Powerball Data Loader" script is saved as a file
# in the same directory, for example, 'powerball_loader.py'.

# You can adjust the import statement based on your file name.
# For this example, we'll assume the file is named 'powerball_loader'.
import data_handler
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

        # You can now add your data analysis or modeling code here.
        # For example, let's print the columns to see what's available.
        print("\nAvailable columns:")
        print(powerball_df.columns.tolist())
        
        # You can access a specific column like this:
        # winning_numbers = powerball_df['Winning Numbers']
        # print("\nFirst 5 winning numbers:")
        # print(winning_numbers.head())
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")

if __name__ == "__main__":
    main()
