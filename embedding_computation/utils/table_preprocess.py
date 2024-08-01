import pandas as pd
import random
import string
import argparse
import os


def make_headers_null(df):
    df.columns = [""] * len(df.columns)
    return df

def determine_delimiter(filepath):
     # Determine delimiter based on file content
        with open(filepath, 'r') as file:
            first_line = file.readline()
            if ';' in first_line:
                delimiter = ';'
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ','
        return delimiter