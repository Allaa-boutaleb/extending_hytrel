"""
Table Sampling Utilities

This module provides various functions for processing and sampling pandas DataFrames
in the context of table data analysis and preparation.
"""

import math
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
MAX_NULLS = 2  # Maximum number of null values allowed in a row
RANDOM_SEED = 42  # Seed for reproducibility in random operations

def remove_rows_with_nulls(df: pd.DataFrame, max_nulls: int = MAX_NULLS) -> pd.DataFrame:
    """
    Remove rows with more than the specified number of null values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        max_nulls (int): Maximum number of null values allowed in a row.

    Returns:
        pd.DataFrame: DataFrame with rows removed based on null count.
    """
    null_counts = df.isnull().sum(axis=1)
    return df[null_counts <= max_nulls]

def shuffle_rows(df: pd.DataFrame, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Shuffle the rows of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        random_state (int): Seed for the random number generator.

    Returns:
        pd.DataFrame: DataFrame with shuffled rows.
    """
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

def prioritize_non_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows based on the number of non-null values, prioritizing rows with fewer nulls.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Sorted DataFrame with rows containing fewer nulls at the top.
    """
    df['null_count'] = df.isnull().sum(axis=1)
    sorted_df = df.sort_values(by='null_count').drop(columns='null_count')
    return sorted_df

def sort_by_tfidf(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    """
    Sort rows based on TF-IDF scores of the combined text in each row.

    Args:
        df (pd.DataFrame): Input DataFrame.
        dropna (bool): Whether to drop NA values before computing TF-IDF.

    Returns:
        pd.DataFrame: DataFrame sorted by TF-IDF scores.
    """
    if dropna:
        df['combined_text'] = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    else:
        df['combined_text'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    tfidf_sum = tfidf_matrix.sum(axis=1).A1
    df['tfidf_score'] = tfidf_sum
    
    sorted_df = df.sort_values(by='tfidf_score', ascending=False)
    return sorted_df.drop(columns=['combined_text', 'tfidf_score'])

def sort_columns_by_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort each column of the DataFrame independently based on its values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with each column sorted independently.
    """
    return pd.DataFrame({col: df[col].sort_values(na_position='last').values for col in df.columns})

def sort_columns_by_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort each column independently based on TF-IDF scores of its values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns sorted by TF-IDF scores.
    """
    def average_tfidf(val):
        if pd.isna(val):
            return -1
        words = str(val).lower().split()
        return sum(tfidf_scores.get(word, 0) for word in words) / len(words) if words else 0

    all_text = pd.concat([df[col].dropna().astype(str) for col in df.columns])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    return pd.DataFrame({col: sorted(df[col], key=average_tfidf, reverse=True) for col in df.columns})

def sample_rows(df: pd.DataFrame, n: int = 30, method: str = 'top') -> pd.DataFrame:
    """
    Sample rows from the DataFrame using various methods.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n (int): Number of rows to sample.
        method (str): Sampling method ('top', 'random', or 'systematic').

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    if method == 'top':
        return df.head(n)
    elif method == 'random':
        return df.sample(n=min(n, len(df)), random_state=RANDOM_SEED)
    elif method == 'systematic':
        rate = math.floor(len(df) / n)
        return df[::rate]
    else:
        raise ValueError("Invalid sampling method. Choose 'top', 'random', or 'systematic'.")

def sample_distinct_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new DataFrame with distinct values from each column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing distinct values from each column.
    """
    distinct_values = {col: df[col].unique() for col in df.columns}
    max_length = max(len(values) for values in distinct_values.values())
    
    padded_values = {
        col: np.pad(values, (0, max_length - len(values)), 
                    mode='constant', constant_values=np.nan)
        for col, values in distinct_values.items()
    }
    
    return pd.DataFrame(padded_values)

def process_table(df: pd.DataFrame, 
                  process: Optional[str] = None, 
                  sample_size: int = 30, 
                  sample_method: str = 'top') -> pd.DataFrame:
    """
    Apply various processing and sampling techniques to a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        process (str, optional): Processing method to apply.
        sample_size (int): Number of rows to sample after processing.
        sample_method (str): Method to use for sampling rows.

    Returns:
        pd.DataFrame: Processed and sampled DataFrame.
    """
    if process == 'remove_nulls':
        df = remove_rows_with_nulls(df)
    elif process == 'prioritize_non_null':
        df = prioritize_non_null_rows(df)
    elif process == 'remove_nulls_and_shuffle':
        df = shuffle_rows(remove_rows_with_nulls(df))
    elif process == 'sort_by_tfidf':
        df = sort_by_tfidf(df)
    elif process == 'sort_by_tfidf_dropna':
        df = sort_by_tfidf(df, dropna=True)
    elif process == 'sort_columns_by_value':
        df = sort_columns_by_value(df)
    elif process == 'sort_columns_by_tfidf':
        df = sort_columns_by_tfidf(df)
    elif process == 'sample_distinct':
        return sample_distinct_values(df)
    
    return sample_rows(df, n=sample_size, method=sample_method)