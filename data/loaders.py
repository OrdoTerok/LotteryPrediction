
"""
data.loaders
------------
Functions for loading raw lottery data from CSV files or APIs (e.g., Data.gov, Kaggle).
Functions:
    - load_csv: Load a CSV file into a DataFrame.
    - fetch_data_from_datagov: Fetch and process data from the Data.gov API.
    - load_data_from_kaggle: Load and process data from a Kaggle CSV file.
    - get_powerball_data: (if present) Load Powerball data from various sources.
"""

import pandas as pd
import requests
import os
import datetime
import logging
logger = logging.getLogger(__name__)

def load_csv(path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(path)

def fetch_data_from_datagov(api_url):
    """
    Fetch and process lottery data from the Data.gov API.

    Parameters
    ----------
    api_url : str
        API endpoint URL for Data.gov.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns ['Draw Date', 'Winning Numbers', 'Multiplier'].
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

def load_data_from_kaggle(file_path):
    """
    Load and process lottery data from a Kaggle CSV file.

    Parameters
    ----------
    file_path : str
        Path to the Kaggle CSV file.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns ['Draw Date', 'Winning Numbers', 'Powerball'].
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

def get_powerball_data(api_url, csv_path):
    """
    Get Powerball data, downloading from Data.gov if today is a draw day, otherwise loading from CSV.

    Parameters
    ----------
    api_url : str
        API endpoint URL for Data.gov.
    csv_path : str
        Path to the local CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with Powerball data.
    """
    today = datetime.datetime.now().date()
    weekday = today.weekday()
    download_today = (weekday == 3) or (weekday == 6)
    if download_today:
        logger.info("Today is the day after a Powerball draw. Downloading fresh data from Data.gov...")
        df = fetch_data_from_datagov(api_url)
        if not df.empty:
            df.to_csv(csv_path, index=False)
        return df
    else:
        logger.info("Not the day after a draw. Using saved data from CSV.")
        return load_data_from_kaggle(csv_path)
# data/loaders.py
# Functions for loading raw data from files or APIs.

import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

# Add more loaders as needed (e.g., for APIs)
