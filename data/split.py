
"""
data.split
----------
Functions for splitting data into training, testing, and validation sets.
Functions:
    - split_dataframe_by_percentage: Split a DataFrame by a percentage.
    - split_train_test: Split a DataFrame into train and test sets using sklearn.
"""

from sklearn.model_selection import train_test_split

def split_dataframe_by_percentage(df, percentage, random_state=None):
    """
    Split a DataFrame into two parts by a given percentage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    percentage : float
        Fraction of data to include in the first split (between 0.0 and 1.0).
    random_state : int or None, optional
        Random seed for reproducibility (default None).

    Returns
    -------
    df1 : pd.DataFrame
        First split DataFrame.
    df2 : pd.DataFrame
        Second split DataFrame.
    """
    if not 0.0 <= percentage <= 1.0:
        raise ValueError("Percentage must be a float between 0.0 and 1.0.")
    df1 = df.sample(frac=percentage, random_state=random_state)
    df2 = df.drop(df1.index)
    return df1, df2
# data/split.py
# Functions for splitting data into train/test/validation sets.

from sklearn.model_selection import train_test_split

def split_train_test(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train and test sets using sklearn's train_test_split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    test_size : float, optional
        Fraction of data to use as test set (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    train : pd.DataFrame
        Training set.
    test : pd.DataFrame
        Test set.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)
