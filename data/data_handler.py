import pandas as pd
import numpy as np
import requests
import io
from typing import Tuple

def download_powerball_data_from_kaggle():
    """
    Downloads the Powerball dataset from the specified Kaggle data URL
    and loads it into a Pandas DataFrame.
    
    This function uses a known, direct URL to a CSV file from a Kaggle
    dataset to ensure reliable access.
    
    Returns:
        pandas.DataFrame or None: A DataFrame containing the Powerball data
                                  if the download is successful, otherwise None.
    """
    # Direct URL to the raw CSV data on a public mirror of the Kaggle dataset.
    # This URL is stable and provides the full historical data.
    url = "https://raw.githubusercontent.com/datasets/powerball/master/data/powerball_usa.csv"
    
    print("Attempting to download data from the Kaggle dataset...")
    try:
        # Use requests to fetch the content from the URL.
        response = requests.get(url, timeout=10)
        # Check if the HTTP request was successful.
        response.raise_for_status() 
        
        # Read the raw text content into an in-memory file-like object.
        # This allows Pandas to read the content directly from the string.
        data = io.StringIO(response.text)
        
        # Use pandas to read the CSV data into a DataFrame.
        df = pd.read_csv(data)
        
        print("Data downloaded and loaded successfully!")
        
        # Print key information about the DataFrame for a quick check.
        print("\nDataFrame Info:")
        df.info()
        print("\nFirst 5 rows:")
        print(df.head())
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching the URL: {e}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        return None

def save_to_parquet(df):
    """
    Saves the given DataFrame to a Parquet file.
    
    Args:
        df (pandas.DataFrame): The DataFrame to save.
        file_path (str): The path where the Parquet file will be saved.
    """
    try:
        df.to_parquet("base_dataset.parquet", index=False)
        print(f"DataFrame successfully saved to base_dataset.parquet")
    except Exception as e:
        print(f"Error saving DataFrame to Parquet: {e}")

def split_dataframe_by_percentage(
    df: pd.DataFrame,
    percentage: float,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a pandas DataFrame into two sub-DataFrames based on a given percentage.

    The split is performed by randomly sampling rows from the source DataFrame
    to create the first sub-DataFrame. The remaining rows are then used to
    create the second sub-DataFrame.

    Args:
        df (pd.DataFrame): The source DataFrame to be split.
        percentage (float): The percentage of the source data to be included
                            in the first DataFrame. This value should be between
                            0.0 and 1.0.
        random_state (int, optional): A seed for the random number generator.
                                      Setting this ensures that the split is
                                      reproducible. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two new DataFrames.
                                           The first DataFrame contains the
                                           percentage-based split, and the second
                                           contains the remaining data.
    """
    if not 0.0 <= percentage <= 1.0:
        raise ValueError("Percentage must be a float between 0.0 and 1.0.")

    # Sample the first DataFrame based on the given percentage (fraction).
    # `random_state` ensures the split is the same every time the function is run.
    df1 = df.sample(frac=percentage, random_state=random_state)

    # The second DataFrame is created by dropping the rows from the first one.
    # This is an efficient way to get all the remaining rows.
    df2 = df.drop(df1.index)

    return df1, df2

def prepare_data_for_lstm(df: pd.DataFrame, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares time series data for an LSTM model.

    This function creates sequences of data (the 'look_back' window) as features
    (X) and the next data point as the target (y). This is the standard format
    for training a time series model.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data. It
                           is expected to have a 'Winning Numbers' column.
        look_back (int): The number of previous data points to use as input
                         features for predicting the next point.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the prepared features (X)
                                       and targets (y).
    """
    df = df.sort_values(by='Draw Date')
    winning_numbers = df['Winning Numbers'].str.split().apply(lambda x: [int(i) for i in x]).values
    
    X, y = [], []
    for i in range(len(winning_numbers) - look_back):
        # Create a sequence of 'look_back' winning number sets.
        X.append(np.concatenate(winning_numbers[i:(i + look_back)]))
        # The target is the very next winning number set.
        y.append(winning_numbers[i + look_back])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Call the main function to perform the download and load.
    powerball_df = download_powerball_data_from_kaggle()
    
    if powerball_df is not None:
        # Now that you have the data, you can start your analysis here.
        print("\nReady for data analysis!")
        print(f"Dataset contains {powerball_df.shape[0]} rows and {powerball_df.shape[1]} columns.")
