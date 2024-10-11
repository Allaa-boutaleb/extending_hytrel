import pandas as pd
import math
from pandas.api.types import infer_dtype
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer


def rows_remove_nulls(df): ## remove rows with null values 
    new_df = df.dropna()
    return new_df

def row_shuffle(df, random_state=42): ##shuffle rows with random seed 
    new_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return new_df

def rows_remove_nulls_max(df, max_nulls=2): ## remove rows with null values
    null_counts = df.isnull().sum(axis=1)
    new_df = df[null_counts <= 2]
    return new_df

def priortize_rows_with_values(df):
    df['null_count'] = df.isnull().sum(axis=1)
    new_df = df.sort_values(by='null_count').drop(columns='null_count')
    return new_df

def row_sort_by_tfidf(df,dropna=False): ##sort rows by tfidf score
    # Combine all text to fit the TF-IDF vectorizer
    if dropna:
        df['combined_text'] = df.apply(lambda row: ' '.join(row.dropna().values.astype(str)), axis=1)
    else: 
        df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    tfidf_sum = tfidf_matrix.sum(axis=1).A1  
    df['tfidf_score'] = tfidf_sum
    new_df = df.sort_values(by='tfidf_score', ascending=False).drop(columns=['combined_text', 'tfidf_score'])
    return new_df 

def value_sort_columns(df):
    sorted_df = pd.DataFrame()
    for col in df.columns:
        sorted_df[col] = df[col].sort_values(na_position='last').reset_index(drop=True)
    return sorted_df

def col_tfidf_sort(df): 

    def average_tfidf(val):
        if pd.isna(val): ##ignore any nan value and assign it -1 to push it down the column 
            return -1
        else: 
            words = str(val).lower().split()
            if len(words) >= 1:
                return sum(tfidf_scores.get(word, 0) for word in words) / len(words) 
            else:
                return 0
    
    # Combine all text to fit the TF-IDF vectorizer and drop the nan values 
    all_text = pd.concat([df[col].dropna().astype(str) for col in df.columns])

    # Initialize and fit the vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    # Apply the sorting based on TF-IDF scores independently for each column

    for column in df.columns:
        values = df[column]
        sorted_values = sorted(values, key=average_tfidf, reverse=True)
        df[column] = sorted_values
    return df

def get_top_rows(df,n=30): ##get top n rows
    return df.head(n)

def pandas_sample(df,n=30):
    if len(df) >= n: 
        return df.sample(n=n, random_state=42)
    else: 
        return df

def pandas_rate_sample(df,n=30):
    rate = math.floor(len(df)/n)
    return df[::rate]

def sample_columns_distinct(df):
    distinct_values = {}
    max_length = 0
    
    for column in df.columns:
        unique_values = df[column].unique()
        distinct_values[column] = unique_values
        max_length = max(max_length, len(unique_values))
    
    for column, values in distinct_values.items():
        if len(values) < max_length:
            distinct_values[column] = list(values) + [float('nan')] * (max_length - len(values))
    
    distinct_df = pd.DataFrame(distinct_values)
    
    return distinct_df