
import pandas as pd
import os
import re
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import sys
import argparse

def get_df(filepath):
    delimiter = determine_delimiter(filepath)
    try:
        df = pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8',on_bad_lines='skip',low_memory=False)
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: Unable to read '{filepath}' using UTF-8 encoding.")
        print(f"Attempting to read '{filepath}' using 'latin1' encoding...")
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, encoding='latin1',on_bad_lines='skip',low_memory=False)
        except Exception as e:
            print(f"Error reading '{filepath}' using 'latin1' encoding: {e}")

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

def extract_entities(text,nlp):
    if isinstance(text, str) and text.strip():
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    return []

def entity_counter(directory_path, benchmark_name,result_path):
    os.makedirs(result_path, exist_ok=True) 
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    nlp = spacy.load('en_core_web_sm')
    entity_counter = Counter()  
    for file in tqdm(all_files):
        df = get_df(file)
        df = df.dropna()
        for text in df.values.flatten():
            entities = extract_entities(text,nlp)
            entity_counter.update(entities)   
    with open(os.path.join(result_path,f'{benchmark_name}_entity_counter.pkl'), 'wb') as f:
        pickle.dump(entity_counter, f)
    return entity_counter

def plot_entity_frequency(nentities,entity_counter, benchmark_name, result_path):
    os.makedirs(result_path, exist_ok=True) 
    common_entities = entity_counter.most_common(nentities)
    common_entity_df = pd.DataFrame(common_entities, columns=['Entity', 'Count'])
    common_entity_df[['Text', 'Label']] = pd.DataFrame(common_entity_df['Entity'].tolist(), index=common_entity_df.index)
    common_entity_df = common_entity_df.drop(columns='Entity')
    label_sums = common_entity_df.groupby('Label')['Count'].sum().reset_index()
    plt.figure(figsize=(14, 10))
    sns.barplot(data=label_sums, x='Count', y='Label')
    plt.title('Summed Counts by Entity Label')
    plt.xlabel('Count')
    plt.ylabel('Entity Labels')
    plt.savefig(os.path.join(result_path,f'summed_counts_{benchmark_name}.png'))
    label_sums = common_entity_df.groupby('Label')['Text'].count().reset_index()
    plt.figure(figsize=(14, 10))
    sns.barplot(data=label_sums, x='Text', y='Label')
    plt.title('Summed Counts by Distinct Entities')
    plt.xlabel('Count')
    plt.ylabel('Entities')
    plt.savefig(os.path.join(result_path,f'summed_counts_distinct_{benchmark_name}.png'))

def main():
    parser = argparse.ArgumentParser(description='Entity Extraction')
    parser.add_argument('--benchmark_name', help='benchmark name', type=str)
    args = parser.parse_args()
    benchmark_name = args.benchmark_name
    paths_dict = { ### place the directories and benchmark names here
    'nextiajd_small':'nextiajd/testbedS/datasets',
    'nextiajd_medium':'nextiajd/testbedM/datasets',
    'webtable_small_var1':'lakebench/datalake/webtable/data_ssd/webtable/small_var1',
    'webtable_small_var2':'lakebench/datalake/webtable/data_ssd/webtable/small_var2'
    }
    result_path = 'entity_results'
    directories = [paths_dict[benchmark_name]]
    benchmark_names = [benchmark_name]
    # print(directories)
    print('extracting entities for:',benchmark_names)
    print(f'paths: {directories}')

    for directory,benchmark_name in tqdm(zip(directories,benchmark_names)):
        result = os.path.join(result_path,benchmark_name)
        entities_counter = entity_counter(directory, benchmark_name,result)
        nentities = 50
        plot_entity_frequency(nentities,entities_counter, benchmark_name, result)  

if __name__ == "__main__":
    main()
