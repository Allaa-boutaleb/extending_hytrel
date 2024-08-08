import argparse
import pandas as pd
import os
# import nltk
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import re
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords

def determine_delimiter(filepath):
        with open(filepath, 'r') as file:
            first_line = file.readline()
            if ';' in first_line:
                delimiter = ';'
            elif ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = None
            return delimiter

def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if not re.search(r'\d', word) and word not in cachedStopWords]
        return words
    return []

def count_words(directory_path, benchmark_name,result_path):
    os.makedirs(result_path, exist_ok=True) ## check if path exists, if not create it
    # Load all CSV files into a single DataFrame
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Initialize a counter
    word_counter = Counter()
    for file in all_files:
        ## read the csv file
        if 'nextiajd' in directory_path:
            delimiter=determine_delimiter(file)
            try:
                df = pd.read_csv(file, delimiter=delimiter)
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: Failed to read {file} with encoding. Trying a different encoding.")
                df = pd.read_csv(file, delimiter=delimiter, encoding='latin1')
            except pd.errors.ParserError as e:
                df = pd.read_csv(file, delimiter=delimiter, on_bad_lines='skip')
        else:
            df = pd.read_csv(file)
        df = df.dropna()
        for text in df.values.flatten():
            words = preprocess_text(text)
            word_counter.update(words)
    # Save the word counter to a pickle file
    with open(os.path.join(result_path,f'{benchmark_name}_word_counter.pkl'), 'wb') as f:
        pickle.dump(word_counter, f)
    return word_counter

def plot_word_frequency(nwords,word_counter, benchmark_name, result_path):
    os.makedirs(result_path, exist_ok=True) ## check if path exists, if not create it
    common_words = word_counter.most_common(nwords)

    # Convert the common words to a DataFrame for visualization
    common_words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
    plt.figure(figsize=(12, 8))
    sns.barplot(data=common_words_df, x='Count', y='Word')
    plt.title(f'Top {nwords} Most Common Words in {benchmark_name}')
    plt.xlabel('Counts')
    plt.ylabel('Words')
    plt.savefig(os.path.join(result_path,f'top_words_{benchmark_name}.png'))

def main():
    global cachedStopWords 
    cachedStopWords = stopwords.words("english")
    parser = argparse.ArgumentParser(description='benchmark stats.')
    parser.add_argument('stat_type', choices=['union_benchmarks', 'join_benchmarks','query_columns','query_table'], help='Type of statistic to collect')
    args = parser.parse_args()
    stat_type = args.stat_type
    if stat_type == 'union_benchmarks':
        ### place the directories and benchmark names here
        directories = ['pylon/data/pylon_benchmark/source','starmie/data/table-union-search-benchmark/small/benchmark','starmie/data/table-union-search-benchmark/large/benchmark','starmie/data/santos/datalake']
        benchmark_names = ['pylon','tus_small','tus_large','santos']
    elif stat_type == 'join_benchmarks':
        ### place the directories and benchmark names here
        directories=['nextiajd/testbedS/datasets','nextiajd/testbedM/datasets','lakebench/datalake/webtable/data_ssd/webtable/small_var1','lakebench/datalake/webtable/data_ssd/webtable/small_var2']
        benchmark_names=['nextiajd_small','nextiajd_medium','webtable_sampled_var1','webtable_sampled_var2']
    result_path = 'bow_results'
    for directory,benchmark_name in tqdm(zip(directories,benchmark_names)):
        result = os.path.join(result_path,benchmark_name)
        word_counter = count_words(directory, benchmark_name,result)
        nwords = 50
        plot_word_frequency(nwords,word_counter, benchmark_name, result)  

if __name__ == "__main__":
    main()