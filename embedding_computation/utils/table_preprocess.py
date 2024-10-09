"""
Table Preprocessing Utilities

This module provides utility functions for preprocessing tabular data,
including handling headers and determining file delimiters.
"""

from typing import List
import pandas as pd

def make_headers_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all column headers with empty strings.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with null (empty string) headers.
    """
    df.columns = [""] * len(df.columns)
    return df

def determine_delimiter(filepath: str) -> str:
    """
    Determine the delimiter used in a CSV-like file based on its content.

    Args:
        filepath (str): Path to the input file.

    Returns:
        str: Detected delimiter (';', '\t', or ',').
    """
    with open(filepath, 'r') as file:
        first_line = file.readline()
        if ';' in first_line:
            return ';'
        elif '\t' in first_line:
            return '\t'
        else:
            return ','


