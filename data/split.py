# data/split.py
# Functions for splitting data into train/test/validation sets.

from sklearn.model_selection import train_test_split

def split_dataframe_by_percentage(df, percentage, random_state=None):
    if not 0.0 <= percentage <= 1.0:
        raise ValueError("Percentage must be a float between 0.0 and 1.0.")
    df1 = df.sample(frac=percentage, random_state=random_state)
    df2 = df.drop(df1.index)
    return df1, df2
# data/split.py
# Functions for splitting data into train/test/validation sets.

from sklearn.model_selection import train_test_split

def split_train_test(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)
