import numpy as np
import pandas as pd 
import os
import sys  
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import psutil
from loguru import logger
from typing import List, Dict

def get_df(filepath):
    delimiter = determine_delimiter(filepath)
    try:
    # Try reading the CSV file with UTF-8 encoding
        df = pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8',on_bad_lines='skip',low_memory=False)
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: Unable to read '{filepath}' using UTF-8 encoding.")
        print(f"Attempting to read '{filepath}' using 'latin1' encoding...")
        try:
            # Try reading the CSV file with 'latin1' encoding
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
            elif '|' in first_line:
                delimiter = '|'
            else:
                delimiter = ','
        return delimiter

def get_join_key(table):
    catalog_path = '../starmie/santos_benchmark/datalake/catalog'
    filename = table.split('.')[0] + '.json'
    path = os.path.join(catalog_path, filename)
    with open(path) as f:
        data = json.load(f)
        return data['primary_key']['column']
    
def get_column_type(table,column):
    catalog_path = '../nextiajd/testbedS/catalog'
    filename = table.split('.')[0] + '.json'
    path = os.path.join(catalog_path, filename)
    with open(path) as f:
        data = json.load(f)
        return data[column]['column_type']


def save_metrics(used_k: List[int], precision_array: List[List[float]], recall_array: List[List[float]],
                 r_precision_array: List[List[float]], fbeta_array: List[List[float]],
                 recall_at_k: List[List[float]], precision_at_k: List[List[float]],
                 map_array: List[List[float]], record_at_k_all: List[Dict[str, Dict[str, float]]],
                 ap_at_k_all: List[Dict[str, float]], run_id: List[str], directory: str, step: int = 1):

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Precision plot
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(run_id):
        plt.plot(used_k, precision_array[i], marker='o', label=method)
    plt.title('Precision at Different k')
    plt.xlabel('k')
    plt.ylabel('P@k')
    plt.legend(loc='best')
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    plt.grid(True)
    plt.savefig(os.path.join(directory, "precision_plot.png"))
    plt.close()

    # Recall plot
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(run_id):
        plt.plot(used_k, recall_array[i], marker='^', label=method)
    plt.title('Recall at Different k')
    plt.xlabel('k')
    plt.ylabel('R@k')
    plt.legend(loc='best')
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    plt.grid(True)
    plt.savefig(os.path.join(directory, "recall_plot.png"))
    plt.close()

    # R-Precision plot
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(run_id):
        plt.plot(used_k, r_precision_array[i], marker='^', label=method)
    plt.title('R-Precision at Different k')
    plt.xlabel('k')
    plt.ylabel('R-Precision@k')
    plt.legend(loc='best')
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    plt.grid(True)
    plt.savefig(os.path.join(directory, "r_precision_plot.png"))
    plt.close()

    # F-beta plot
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(run_id):
        plt.plot(used_k, fbeta_array[i], marker='^', label=method)
    plt.title('F-beta at Different k')
    plt.xlabel('k')
    plt.ylabel('F-beta@k')
    plt.legend(loc='best')
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    plt.grid(True)
    plt.savefig(os.path.join(directory, "fbeta_plot.png"))
    plt.close()

    # Recall and Precision distribution plots
    for i, method in enumerate(run_id):
        plt.figure(figsize=(10, 6))
        plt.hist(recall_at_k[i], bins=10, edgecolor='black', linewidth=1.2)
        plt.xlabel('Recall')
        plt.ylabel('Frequency')
        plt.title(f'Recall Distribution - {method}')
        plt.savefig(os.path.join(directory, f"recall_distribution_at_k_plot_{method}.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(precision_at_k[i], bins=10, edgecolor='black', linewidth=1.2)
        plt.xlabel('Precision')
        plt.ylabel('Frequency')
        plt.title(f'Precision Distribution - {method}')
        plt.savefig(os.path.join(directory, f"precision_distribution_at_k_plot_{method}.png"))
        plt.close()

    # MAP plot
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(run_id):
        plt.plot(used_k, map_array[i], marker='o', label=method)
    plt.title('MAP at Different k')
    plt.xlabel('k')
    plt.ylabel('MAP@k')
    plt.legend(loc='best')
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    plt.grid(True)
    plt.savefig(os.path.join(directory, "map_plot_all.png"))
    plt.close()

    # Individual MAP plots
    for i, method in enumerate(run_id):
        plt.figure(figsize=(10, 6))
        plt.plot(used_k, map_array[i], marker='o', label=method)
        plt.title(f'MAP at Different k - {method}')
        plt.xlabel('k')
        plt.ylabel('MAP@k')
        plt.legend(loc='best')
        if max(used_k) <= 10:
            plt.xticks(used_k, [int(k) for k in used_k])
        plt.grid(True)
        plt.savefig(os.path.join(directory, f"map_plot_{method}.png"))
        plt.close()

    # Save summary Excel files
    for i, method in enumerate(run_id):
        summary_data = {
            'k': used_k,
            'Precision': precision_array[i],
            'Recall': recall_array[i],
            'r-Precision': r_precision_array[i],
            'F-beta': fbeta_array[i],
            'MAP': map_array[i]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(os.path.join(directory, f"precision_recall_summary_{method}.xlsx"), index=False)

    # Save query-wise metrics
    for i, method in enumerate(run_id):
        queries = []
        precision_dict = {k: [] for k in used_k}
        recall_dict = {k: [] for k in used_k}
        ap_dict = {k: [] for k in used_k}
        
        for q in record_at_k_all[i].keys():
            queries.append(q)
            for k in used_k:
                precision_dict[k].append(record_at_k_all[i][q][k]['precision'])
                recall_dict[k].append(record_at_k_all[i][q][k]['recall'])
                ap_dict[k].append(ap_at_k_all[i][q][k])

        query_data = {'Query': queries}
        for k in used_k:
            query_data[f'P@{k}'] = precision_dict[k]
        for k in used_k:    
            query_data[f'R@{k}'] = recall_dict[k]
        for k in used_k:
            query_data[f'AP@{k}'] = ap_dict[k]
        
        df_query = pd.DataFrame(query_data)
        df_query.to_excel(os.path.join(directory, f"queries_at_k_summary_{method}.xlsx"), index=False)

    print(f"All metrics and plots saved in {directory}")
    logger.warning(f"used_k: {used_k}")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB