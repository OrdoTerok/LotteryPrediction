import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple
import datetime
import logging
logger = logging.getLogger(__name__)


class DataHandler:
    @staticmethod
    def fetch_data_from_datagov(api_url):
        """
        Fetches Powerball data from the Data.gov API.
        Args:
            api_url (str): The URL of the Data.gov Powerball API.
        Returns:
            pd.DataFrame: A DataFrame containing the fetched data, or an empty DataFrame if the request fails.
        """
        logger.info("Fetching data from Data.gov API...")
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            if not df.empty:
                df = df[['draw_date', 'winning_numbers', 'multiplier']]
                df.rename(columns={'draw_date': 'Draw Date', 'winning_numbers': 'Winning Numbers', 'multiplier': 'Multiplier'}, inplace=True)
                logger.info("Successfully fetched and processed Data.gov data.")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Data.gov: {e}")
            return pd.DataFrame()


    @staticmethod
    def load_data_from_kaggle(file_path):
        """
        Loads Powerball data from a local Kaggle CSV file.
        Args:
            file_path (str): The path to the Kaggle CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data, or an empty DataFrame if the file is not found.
        """
        logger.info(f"Attempting to load data from local file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Error: The file '{file_path}' was not found.")
            logger.error("Please download the CSV from Kaggle and place it in the same directory.")
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                df = df[['Date', 'Winning Numbers', 'Powerball']]
                df.rename(columns={'Date': 'Draw Date'}, inplace=True)
                df.columns = ['Draw Date', 'Winning Numbers', 'Powerball']
                logger.info("Successfully loaded and processed Kaggle data.")
            return df
        except Exception as e:
            logger.error(f"Error reading the CSV file: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_powerball_data(api_url, csv_path):
        """
        Only download from data.gov the day after a draw (Thursday or Sunday). Otherwise, use the saved CSV.
        """
        today = datetime.datetime.now().date()
        weekday = today.weekday()  # Monday=0, Sunday=6
        download_today = (weekday == 3) or (weekday == 6)
        if download_today:
            logger.info("Today is the day after a Powerball draw. Downloading fresh data from Data.gov...")
            df = DataHandler.fetch_data_from_datagov(api_url)
            if not df.empty:
                df.to_csv(csv_path, index=False)
            return df
        else:
            logger.info("Not the day after a draw. Using saved data from CSV.")
            return DataHandler.load_data_from_kaggle(csv_path)

    @staticmethod
    def combine_and_clean_data(df_datagov, df_kaggle):
        """
        Combines the two DataFrames, removes duplicates, and standardizes data types.
        Args:
            df_datagov (pd.DataFrame): DataFrame from Data.gov.
            df_kaggle (pd.DataFrame): DataFrame from Kaggle.
        Returns:
            pd.DataFrame: A single, cleaned DataFrame with combined data.
        """
        logger.info("Combining datasets...")
        combined_df = pd.concat([df_datagov, df_kaggle], ignore_index=True)
        combined_df.drop_duplicates(subset=['Draw Date', 'Winning Numbers', 'Multiplier'], inplace=True)
        combined_df['Draw Date'] = pd.to_datetime(combined_df['Draw Date'])
        combined_df.sort_values(by='Draw Date', ascending=True, inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        logger.info(f"Combined dataset contains {len(combined_df)} unique records.")
        return combined_df

    @staticmethod
    def save_to_file(df):
        """
        Saves the given DataFrame to a CSV file.
        Args:
            df (pandas.DataFrame): The DataFrame to save.
        """
        DataHandler._save_df_with_logging(df, "data_sets/base_dataset.csv")

    @staticmethod
    def _save_df_with_logging(df, path):
        try:
            df.to_csv(path, index=False)
            logger.info(f"DataFrame successfully saved to {path}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to {path}: {e}")

    @staticmethod
    def split_dataframe_by_percentage(
        df: pd.DataFrame,
        percentage: float,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a pandas DataFrame into two sub-DataFrames based on a given percentage.
        Args:
            df (pd.DataFrame): The source DataFrame to be split.
            percentage (float): The percentage of the source data to be included in the first DataFrame. This value should be between 0.0 and 1.0.
            random_state (int, optional): A seed for the random number generator. Setting this ensures that the split is reproducible. Defaults to None.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two new DataFrames. The first DataFrame contains the percentage-based split, and the second contains the remaining data.
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("Percentage must be a float between 0.0 and 1.0.")
        df1 = df.sample(frac=percentage, random_state=random_state)
        df2 = df.drop(df1.index)
        return df1, df2

    @staticmethod
    def prepare_data_for_lstm(df: pd.DataFrame, look_back: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepares time series data for an LSTM model with one-hot encoded targets for categorical crossentropy.
        The first 5 numbers are one-hot encoded with 69 classes (1-69), and the 6th number (Powerball) is one-hot encoded with 26 classes (1-26).
        If config.ITERATIVE_STACKING is True and previous predictions exist, appends them as meta-features.
        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: (X, (y_first_five, y_sixth))
                - X: features (optionally with meta-features)
                - y_first_five: (num_samples, 5, 69) one-hot
                - y_sixth: (num_samples, 1, 26) one-hot
        """
        import config.config as config
        import json
        df = df.sort_values(by='Draw Date')
        num_first = 5
        num_first_classes = 69
        num_sixth_classes = 26
        meta_cols = [col for col in df.columns if col.startswith('prev_pred_ball_') or col == 'prev_pred_sixth' or col == 'is_pseudo']
        # Parse winning numbers into a 2D numpy array
        winning_numbers = np.stack(df['Winning Numbers'].str.split().apply(lambda x: [int(i) for i in x]).values)
        n_rows = winning_numbers.shape[0]
        # Prepare meta-features if present
        meta_arr = None
        if meta_cols:
            meta_arr = df[meta_cols].to_numpy(dtype=np.float32)
        # Build windowed features using stride tricks
        num_samples = n_rows - look_back
        if num_samples <= 0:
            return np.empty((0,)), (np.empty((0,)), np.empty((0,)))
        # Windowed winning numbers: shape (num_samples, look_back, 6)
        wn_windows = np.lib.stride_tricks.sliding_window_view(winning_numbers, window_shape=(look_back, 6))[:, 0, :, :]
        # Windowed meta-features: shape (num_samples, look_back, meta_dim) if meta_cols else None
        if meta_arr is not None:
            meta_windows = np.lib.stride_tricks.sliding_window_view(meta_arr, window_shape=(look_back, meta_arr.shape[1]))[:, 0, :, :]
            # Concatenate base and meta along last axis
            X = np.concatenate([wn_windows, meta_windows], axis=-1)
        else:
            X = wn_windows
        # Prepare targets
        target_numbers = winning_numbers[look_back:]
        # One-hot encode first five
        y_first_five = np.zeros((num_samples, num_first, num_first_classes), dtype=np.float32)
        idx = np.arange(num_samples)[:, None]
        pos = np.arange(num_first)
        vals = target_numbers[:, :num_first] - 1
        mask = (vals >= 0) & (vals < num_first_classes)
        y_first_five[idx, pos, vals] = mask
        # One-hot encode sixth
        y_sixth = np.zeros((num_samples, 1, num_sixth_classes), dtype=np.float32)
        n6 = target_numbers[:, num_first] - 1
        mask6 = (n6 >= 0) & (n6 < num_sixth_classes)
        y_sixth[np.arange(num_samples), 0, n6] = mask6
        return X, (y_first_five, y_sixth)
