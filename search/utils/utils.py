import numpy as np
import pandas as pd 
import os
import sys  
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import psutil

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
def save_metrics(used_k,precision_array,recall_array,r_precision_array, fbeta_array,recall_at_k,precision_at_k,map_array,record_at_k_all,run_id,directory,step=1):

    if not os.path.exists(directory):
        os.makedirs(directory)

    # File names for the saved chart and Excel file
    chart_file_name = "precision_plot.png"
    chart_file_path = os.path.join(directory, chart_file_name)
    # Plotting the precision and recall
    plt.figure(figsize=(10, 6))

    for i in range(len(precision_array)):
        print(i)
        plt.plot(used_k, precision_array[i], marker='o', label=run_id[i])

    # Adding titles and labels
    plt.title('Precision at Different k')
    plt.xlabel('k')
    plt.ylabel('P@k')
    plt.legend(loc=7)
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    plt.grid(True)

    # Save the plot
    plt.savefig(chart_file_path)

    print(f"precision chart saved at {chart_file_path}")
    chart_file_name = "recall_plot.png"
    plt.figure(figsize=(10, 6))
    chart_file_path = os.path.join(directory, chart_file_name)
    for i in range(len(recall_array)):
        plt.plot(used_k, recall_array[i], marker='^', label=run_id[i])
    
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    # Adding titles and labels
    plt.title('Recall at Different k')
    plt.xlabel('k')
    plt.ylabel('R@k')
    plt.legend(loc=7)
    plt.grid(True)

    # Save the plot
    plt.savefig(chart_file_path)
    print(f"recall chart saved at {chart_file_path}")

    ## r-Precision plot
    chart_file_name = "r_precision_plot.png"
    plt.figure(figsize=(10, 6))
    chart_file_path = os.path.join(directory, chart_file_name)
    for i in range(len(r_precision_array)):
        plt.plot(used_k, r_precision_array[i], marker='^', label=run_id[i])
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    # Adding titles and labels
    plt.title('R-Precision at Different k')
    plt.xlabel('k')
    plt.ylabel('R-Precision@k')
    plt.legend(loc=7)
    plt.grid(True)
    plt.savefig(chart_file_path)
    print(f"r-precision chart saved at {chart_file_path}")
    ##fbeta plot
    chart_file_name = "fbeta_plot.png"
    plt.figure(figsize=(10, 6))
    chart_file_path = os.path.join(directory, chart_file_name)
    for i in range(len(fbeta_array)):
        plt.plot(used_k, fbeta_array[i], marker='^', label=run_id[i])
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    # Adding titles and labels
    plt.title('fbeta at Different k')
    plt.xlabel('k')
    plt.ylabel('fbeta@k')
    plt.legend(loc=7)
    plt.grid(True)
    plt.savefig(chart_file_path)
    print(f"fbeta chart saved at {chart_file_path}")
    ## add distribution of precision and recall
    # Plot the recall distribution
    for i in range(len(run_id)):
        chart_file_name = f"recall_distribution_at_k_plot_{run_id[i]}.png"
        plt.figure(figsize=(10, 6))
        chart_file_path = os.path.join(directory, chart_file_name)
        plt.hist(recall_at_k[i], bins=10,edgecolor='black', linewidth=1.2)
        plt.xlabel('Recall')
        plt.ylabel('Frequency')
        plt.title('Recall Distribution')
        plt.savefig(chart_file_path)    
    # Plot the precision distribution
    for i in range(len(run_id)):
        chart_file_name = f"precision_distribution_at_k_plot{run_id[i]}.png"
        plt.figure(figsize=(10, 6))
        chart_file_path = os.path.join(directory, chart_file_name)
        plt.hist(precision_at_k[i], bins=10,edgecolor='black', linewidth=1.2)
        plt.xlabel('Precision')
        plt.ylabel('Frequency')
        plt.title('Precision Distribution')
        plt.savefig(chart_file_path)
    ### plot MAP 
    chart_file_name = "map_plot_all.png"
    plt.figure(figsize=(10, 6))
    chart_file_path = os.path.join(directory, chart_file_name)
    for i in range(len(run_id)):
        plt.plot(used_k, map_array[i], marker='o', label=run_id[i])
    if max(used_k) <= 10:
        plt.xticks(used_k, [int(k) for k in used_k])
    # Adding titles and labels
    plt.title('MAP at Different k')
    plt.xlabel('k')
    plt.ylabel('MAP@k')
    plt.legend(loc=7)
    plt.grid(True)
    plt.savefig(chart_file_path)
    ## plot map indiviually
    for i in range(len(run_id)):
        chart_file_name = f"map_plot_{run_id[i]}.png"
        plt.figure(figsize=(10, 6))
        chart_file_path = os.path.join(directory, chart_file_name)
        plt.plot(used_k, map_array[i], marker='o', label=run_id[i])
        if max(used_k) <= 10:
            plt.xticks(used_k, [int(k) for k in used_k])
        # Adding titles and labels
        plt.title('MAP at Different k')
        plt.xlabel('k')
        plt.ylabel('MAP@k')
        plt.legend(loc=7)
        plt.grid(True)
        plt.savefig(chart_file_path)
    for i in range(len(run_id)):
        excel_file_name = f"precision_recall_summary_{run_id[i]}.xlsx"
        excel_file_path = os.path.join(directory, excel_file_name)
        # Creating a DataFrame for the summary table
        data = {
            'k': used_k,
            'Precision': precision_array[i],
            'Recall': recall_array[i],
            'r-Precision': r_precision_array[i],
            'fbeta': fbeta_array[i],
            'map': map_array[i]
        }
        
        
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        df.to_excel(excel_file_path, index=False)

    print(f"Excel sheet saved at {excel_file_path}")
    ### save the records of precision and recall at k for each query for later analysis 
    for i in range(len(run_id)):
        excel_file_name = f"queries_at_k_summary_{run_id[i]}.xlsx"
        excel_file_path = os.path.join(directory, excel_file_name)
        # Creating a DataFrame for the summary table
        queries = []
        precision = []
        recall = []
        for q in record_at_k_all[i].keys():
            queries.append(q)
            precision.append(record_at_k_all[i][q]['precision'])
            recall.append(record_at_k_all[i][q]['recall'])
        data = {
            'queries': queries,
            'Precision': precision,
            'Recall': recall
        }
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        df.to_excel(excel_file_path, index=False)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB
        