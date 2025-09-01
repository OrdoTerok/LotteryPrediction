# data/preprocessing.py
# Functions for data cleaning, feature engineering, and preprocessing.

import pandas as pd
import numpy as np
import logging

def combine_and_clean_data(df_datagov, df_kaggle):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Combining datasets...")
    combined_df = pd.concat([df_datagov, df_kaggle], ignore_index=True)
    # Remove duplicate entries based on the draw date and winning numbers.
    subset_cols = [col for col in ['Draw Date', 'Winning Numbers', 'Multiplier'] if col in combined_df.columns]
    if not subset_cols:
        logger.error("No columns found for duplicate removal. Returning original combined DataFrame.")
        return combined_df
    combined_df.drop_duplicates(subset=subset_cols, inplace=True)
    if 'Draw Date' in combined_df.columns:
        combined_df['Draw Date'] = pd.to_datetime(combined_df['Draw Date'], errors='coerce')
        combined_df = combined_df.dropna(subset=['Draw Date'])
        combined_df.sort_values(by='Draw Date', ascending=True, inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
    else:
        logger.error("'Draw Date' column missing after combining. Skipping date conversion and sort.")
    logger.info(f"Combined dataset contains {len(combined_df)} unique records.")
    return combined_df

def save_to_file(df, file_path="data_sets/base_dataset.csv"):
    logger = logging.getLogger(__name__)
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV: {e}")

def prepare_data_for_lstm(df: pd.DataFrame, look_back: int):
    import config
    df = df.sort_values(by='Draw Date')
    winning_numbers = df['Winning Numbers'].str.split().apply(lambda x: [int(i) for i in x]).values
    X = []
    y_first_five = []
    y_sixth = []
    num_first = 5
    num_first_classes = 69
    num_sixth_classes = 26
    meta_cols = [col for col in df.columns if col.startswith('prev_pred_ball_') or col == 'prev_pred_sixth' or col == 'is_pseudo']
    for i in range(len(winning_numbers) - look_back):
        window_feats = []
        for j in range(i, i + look_back):
            base = np.array(winning_numbers[j])
            meta = []
            if meta_cols:
                meta_row = df.iloc[j][meta_cols] if j < len(df) else None
                if meta_row is not None:
                    meta = meta_row.values.astype(np.float32)
            if len(meta) > 0:
                window_feats.append(np.concatenate([base, meta]))
            else:
                window_feats.append(base)
        X.append(np.stack(window_feats))
        target_numbers = winning_numbers[i + look_back]
        first_five_onehot = np.zeros((num_first, num_first_classes), dtype=np.float32)
        for j, n in enumerate(target_numbers[:num_first]):
            if 1 <= n <= num_first_classes:
                first_five_onehot[j, n - 1] = 1.0
        y_first_five.append(first_five_onehot)
        sixth_onehot = np.zeros((1, num_sixth_classes), dtype=np.float32)
        n6 = target_numbers[num_first]
        if 1 <= n6 <= num_sixth_classes:
            sixth_onehot[0, n6 - 1] = 1.0
        y_sixth.append(sixth_onehot)
    return np.array(X), (np.array(y_first_five), np.array(y_sixth))
# data/preprocessing.py
# Functions for data cleaning, feature engineering, and preprocessing.

# Example placeholder function
def clean_data(df):
    # Implement cleaning logic here
    return df
