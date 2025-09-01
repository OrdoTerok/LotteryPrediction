# data/loaders.py
# Functions for loading raw data from files or APIs.

import pandas as pd
import requests
import os
import datetime

def load_csv(path):
    return pd.read_csv(path)

def fetch_data_from_datagov(api_url):
    print("Fetching data from Data.gov API...")
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df = df[['draw_date', 'winning_numbers', 'multiplier']]
            df.rename(columns={'draw_date': 'Draw Date', 'winning_numbers': 'Winning Numbers', 'multiplier': 'Multiplier'}, inplace=True)
            print("Successfully fetched and processed Data.gov data.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Data.gov: {e}")
        return pd.DataFrame()

def load_data_from_kaggle(file_path):
    print(f"Attempting to load data from local file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please download the CSV from Kaggle and place it in the same directory.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        if not df.empty:
            df = df[['Date', 'Winning Numbers', 'Powerball']]
            df.rename(columns={'Date': 'Draw Date'}, inplace=True)
            df.columns = ['Draw Date', 'Winning Numbers', 'Powerball']
            print("Successfully loaded and processed Kaggle data.")
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return pd.DataFrame()

def get_powerball_data(api_url, csv_path):
    today = datetime.datetime.now().date()
    weekday = today.weekday()
    download_today = (weekday == 3) or (weekday == 6)
    if download_today:
        print("Today is the day after a Powerball draw. Downloading fresh data from Data.gov...")
        df = fetch_data_from_datagov(api_url)
        if not df.empty:
            df.to_csv(csv_path, index=False)
        return df
    else:
        print("Not the day after a draw. Using saved data from CSV.")
        return load_data_from_kaggle(csv_path)
# data/loaders.py
# Functions for loading raw data from files or APIs.

import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

# Add more loaders as needed (e.g., for APIs)
