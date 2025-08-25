import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple

class DataHandler:
    def fetch_data_from_datagov(api_url):
        """
        Fetches Powerball data from the Data.gov API.

        Args:
            api_url (str): The URL of the Data.gov Powerball API.

        Returns:
            pd.DataFrame: A DataFrame containing the fetched data, or an empty
                          DataFrame if the request fails.
        """
        print("Fetching data from Data.gov API...")
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            df = pd.DataFrame(data)
        
            # Select and rename columns to a consistent format.
            if not df.empty:
                df = df[['draw_date', 'winning_numbers', 'multiplier']]
                df.rename(columns={'draw_date': 'Draw Date', 
                                   'winning_numbers': 'Winning Numbers', 
                                   'multiplier': 'Multiplier'}, inplace=True)
                print("Successfully fetched and processed Data.gov data.")
            return df
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Data.gov: {e}")
            return pd.DataFrame()


    def load_data_from_kaggle(file_path):
        """
        Loads Powerball data from a local Kaggle CSV file.

        Args:
            file_path (str): The path to the Kaggle CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data, or an empty
                          DataFrame if the file is not found.
        """
        print(f"Attempting to load data from local file: {file_path}")
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            print("Please download the CSV from Kaggle and place it in the same directory.")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)

            # Select and rename columns to a consistent format.
            if not df.empty:
                df = df[['Date', 'Winning Numbers', 'Powerball']]
                df.rename(columns={'Date': 'Draw Date'}, inplace=True)
                # Ensure consistent column naming after rename
                df.columns = ['Draw Date', 'Winning Numbers', 'Powerball']
                print("Successfully loaded and processed Kaggle data.")
            return df
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            return pd.DataFrame()

    def combine_and_clean_data(df_datagov, df_kaggle):
        """
        Combines the two DataFrames, removes duplicates, and standardizes data types.

        Args:
            df_datagov (pd.DataFrame): DataFrame from Data.gov.
            df_kaggle (pd.DataFrame): DataFrame from Kaggle.

        Returns:
            pd.DataFrame: A single, cleaned DataFrame with combined data.
        """
        print("Combining datasets...")
    
        # Concatenate the two dataframes.
        combined_df = pd.concat([df_datagov, df_kaggle], ignore_index=True)
    
        # Remove duplicate entries based on the draw date and winning numbers.
        combined_df.drop_duplicates(subset=['Draw Date', 'Winning Numbers', 'Multiplier'], inplace=True)

        # Convert 'Draw Date' column to datetime objects for proper sorting.
        combined_df['Draw Date'] = pd.to_datetime(combined_df['Draw Date'])
    
        # Sort the combined data by date to have the most recent entries at the top.
        combined_df.sort_values(by='Draw Date', ascending=True, inplace=True)
    
        # Reset index after sorting and dropping duplicates.
        combined_df.reset_index(drop=True, inplace=True)

        print(f"Combined dataset contains {len(combined_df)} unique records.")
        return combined_df

    def save_to_file(df):
        """
        Saves the given DataFrame to a Parquet file.
    
        Args:
            df (pandas.DataFrame): The DataFrame to save.
            file_path (str): The path where the Parquet file will be saved.
        """
        try:
            df.to_csv("data_sets/base_dataset.csv", index=False)
            print(f"DataFrame successfully saved to data_sets/base_dataset.csv")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")

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

    def prepare_data_for_lstm(df: pd.DataFrame, look_back: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepares time series data for an LSTM model with one-hot encoded targets for categorical crossentropy.

        The first 5 numbers are one-hot encoded with 69 classes (1-69),
        and the 6th number (Powerball) is one-hot encoded with 26 classes (1-26).

        If config.ITERATIVE_STACKING is True and previous predictions exist, appends them as meta-features.

        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: (X, (y_first_five, y_sixth))
                - X: features (optionally with meta-features)
                - y_first_five: (num_samples, 5, 69) one-hot
                - y_sixth: (num_samples, 1, 26) one-hot
        """
        import config
        import json
        df = df.sort_values(by='Draw Date')
        winning_numbers = df['Winning Numbers'].str.split().apply(lambda x: [int(i) for i in x]).values

        X = []
        y_first_five = []
        y_sixth = []
        meta_features = []
        num_first = 5
        num_first_classes = 69
        num_sixth_classes = 26
        use_meta = getattr(config, 'ITERATIVE_STACKING', False)
        meta_preds = None
        if use_meta and os.path.exists('results_predictions.json'):
            try:
                with open('results_predictions.json', 'r') as f:
                    meta_preds = json.load(f)
                meta_first = np.array(meta_preds.get('first_five_pred_numbers'))
                meta_sixth = np.array(meta_preds.get('sixth_pred_number'))
            except Exception as e:
                print(f"[IterativeStacking] Could not load previous predictions: {e}")
                meta_preds = None
        # Determine meta-feature length for padding
        meta_feat_len = 0
        if meta_preds is not None:
            meta_first_shape = meta_first.shape[1:] if meta_first is not None else (5,)
            meta_sixth_shape = meta_sixth.shape[1:] if meta_sixth is not None else (1,)
            meta_feat_len = np.prod(meta_first_shape) + np.prod(meta_sixth_shape)
        for i in range(len(winning_numbers) - look_back):
            base_feat = np.concatenate(winning_numbers[i:(i + look_back)])
            # Optionally append meta-features from previous predictions
            if meta_preds is not None and i < len(meta_first):
                meta_feat = np.concatenate([
                    meta_first[i].flatten(),
                    meta_sixth[i].flatten()
                ])
                X.append(np.concatenate([base_feat, meta_feat]))
            elif meta_feat_len > 0:
                # Pad with zeros if meta-features are expected but missing
                X.append(np.concatenate([base_feat, np.zeros(meta_feat_len, dtype=np.float32)]))
            else:
                X.append(base_feat)
            target_numbers = winning_numbers[i + look_back]
            # First 5 numbers one-hot (shape: 5, 69)
            first_five_onehot = np.zeros((num_first, num_first_classes), dtype=np.float32)
            for j, n in enumerate(target_numbers[:num_first]):
                if 1 <= n <= num_first_classes:
                    first_five_onehot[j, n - 1] = 1.0
            y_first_five.append(first_five_onehot)
            # Sixth number one-hot (shape: 1, 26)
            sixth_onehot = np.zeros((1, num_sixth_classes), dtype=np.float32)
            n6 = target_numbers[num_first]
            if 1 <= n6 <= num_sixth_classes:
                sixth_onehot[0, n6 - 1] = 1.0
            y_sixth.append(sixth_onehot)
        return np.array(X), (np.array(y_first_five), np.array(y_sixth))
