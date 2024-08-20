

import argparse
import parser
import pickle
import os
import shutil
import random
from tqdm import tqdm
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# import chardet
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def count_nulls(df, column):
    return df[column].isnull().sum()

def count_numeric_cells(df,column): 
#     return df[column].apply(lambda val: any(ch.isdigit() for ch in val)).sum()
    return df[column].apply(lambda val: any(ch.isdigit() for ch in str(val))).sum()

def count_rows(df):
    return len(df)

def count_columns(df):
    return len(df.columns)

def count_column_types(df):
    total_int_columns =0
    total_float_columns=0
    total_object_columns=0
    for dtype in df.dtypes:
        if pd.api.types.is_integer_dtype(dtype):
            total_int_columns += 1
        elif pd.api.types.is_float_dtype(dtype):
            total_float_columns += 1
        elif pd.api.types.is_object_dtype(dtype):
            total_object_columns += 1
    return total_int_columns, total_float_columns, total_object_columns


def get_stats_query_columns(dictionary,directory):
    ''''
    input: dictionary of pairs (table, column) that represent queries, directory is the data directory where the datasets reside 
    returns: number of numeric columns in the dictionary, number of non numeric columns, and sum of nulls values
    '''
    count_numeric_cols = 0
    count_non_numeric_cols = 0
    nulls = 0
    files_count = len(dictionary.keys())
    for key in dictionary.keys(): 
        file = os.path.join(directory,key[0])
        if 'nextiajd' in directory:
            delimiter=determine_delimiter(file)
            try:
                tb = pd.read_csv(file, delimiter=delimiter)
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: Failed to read {file} with encoding. Trying a different encoding.")
                tb = pd.read_csv(file, delimiter=delimiter, encoding='latin1')
            except pd.errors.ParserError as e:
                tb = pd.read_csv(file, delimiter=delimiter, on_bad_lines='skip')
        else:
            tb = pd.read_csv(file)
        print(file)
        numeric_cells = count_numeric_cells(tb,key[1])
        nulls += count_nulls(tb, key[1])
        if numeric_cells > 1:
            count_numeric_cols +=1 
        else:
            count_non_numeric_cols +=1
    return count_numeric_cols,count_non_numeric_cols,nulls

def query_columns_stats(benchmark_names,dictionaries,directories):
    all_count_numeric = []
    all_count_non_numeric = []
    all_null = []
    all_queries = []
    for i in range(len(dictionaries)):
        with open(dictionaries[i],'rb') as f:
            dictionary = pickle.load(f)
        all_queries.append(len(dictionary))
        count_numeric_cols,count_non_numeric_cols,nulls = get_stats_query_columns(dictionary,directories[i])
        all_count_numeric.append(count_numeric_cols)
        all_count_non_numeric.append(count_non_numeric_cols)
        all_null.append(nulls)
    
    data = {'benchmarks':benchmark_names,'path':directories,'ttl_queries':all_queries,'nulls':all_null,'cols_non_numeric':all_count_non_numeric,
            'cols_numeric':all_count_numeric}

    summary_df = pd.DataFrame(data)
    return summary_df

def get_stats_query_tables(dictionary,directory):
    count_numeric_cols = 0
    count_non_numeric_cols = 0
    nulls = 0
    files_count = len(dictionary.keys())
    for key in dictionary.keys(): 
#         if key in queries:
        if '.csv' not in key: 
            key = key+'.csv'
        if key == 'glassdoor_jobs.csv' and 'pylon' in directory:
            key = 'Glassdoor_jobs.csv'
        file = os.path.join(directory,key)
        tb = pd.read_csv(file)
        for column in tb.columns:
            numeric_cells = count_numeric_cells(tb,column)
            nulls += count_nulls(tb,column)

            if numeric_cells > 1:
                count_numeric_cols +=1 
            else:
                count_non_numeric_cols +=1
    return count_numeric_cols,count_non_numeric_cols,nulls

def query_tables_stats(benchmark_names,dictionaries,directories):
    all_count_numeric = []
    all_count_non_numeric = []
    all_null = []
    all_queries = []
    for i in range(len(dictionaries)):
        with open(dictionaries[i],'rb') as f:
            dictionary = pickle.load(f)
#         all_files_in_directory = set(os.listdir(directories[i]))
#         if benchmark_names[i] =='pylon':
#             for q in dictionary.keys(): 
                
#         queries=set(all_files_in_directory).intersection(dictionary.keys())
        all_queries.append(len(dictionary.keys()))
        count_numeric_cols,count_non_numeric_cols,nulls = get_stats_query_tables(dictionary,directories[i])
        all_count_numeric.append(count_numeric_cols)
        all_count_non_numeric.append(count_non_numeric_cols)
        all_null.append(nulls)
    
    data = {'benchmarks':benchmark_names,'path':directories,'ttl_queries':all_queries,'nulls':all_null,'cols_non_numeric':all_count_non_numeric,
            'cols_numeric':all_count_numeric}

    summary_df = pd.DataFrame(data)
    return summary_df

def determine_delimiter(filepath):
#     encoding = detect_encoding(filepath)
#     print(f"Detected encoding: {encoding}")

#     # List of encodings to try if the detected encoding fails
#     encodings_to_try = [encoding, 'utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
#     for enc in encodings_to_try:
#         try:
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
        
def get_stats(dictionary,directory):
    count_numeric_cols = 0
    count_non_numeric_cols = 0
    nulls = 0
    ttl_cols = 0
    ttl_rows = 0
    float_count = 0
    int_count = 0
    object_count = 0
    files_rows = []
    files_cols = []
    files = [os.path.join(directory, f) for f in tqdm(os.listdir(directory)) if os.path.isfile(os.path.join(directory, f)) and f!='.DS_Store']
    files_count = len(files)
    for file in files:
        if 'nextiajd' in directory:
            delimiter=determine_delimiter(file)
            try:
                tb = pd.read_csv(file, delimiter=delimiter)
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: Failed to read {file} with encoding. Trying a different encoding.")
                tb = pd.read_csv(file, delimiter=delimiter, encoding='latin1')
            except pd.errors.ParserError as e:
                tb = pd.read_csv(file, delimiter=delimiter, on_bad_lines='skip')
        else:
            tb = pd.read_csv(file)
        int_columns, float_columns, object_columns = count_column_types(tb)
        float_count += float_columns
        int_count += int_columns
        object_count += object_columns
        cols = count_columns(tb)
        rows = count_rows(tb)
        ttl_cols += cols
        ttl_rows += rows
        files_rows.append(rows)
        files_cols.append(cols)
        for column in tb.columns:
            numeric_cells = count_numeric_cells(tb,column)
            nulls += count_nulls(tb,column)
            if numeric_cells > 1:
                count_numeric_cols +=1 
            else:
                count_non_numeric_cols +=1
    return files_count,count_numeric_cols, count_non_numeric_cols, nulls, ttl_rows, ttl_cols,max(files_rows),min(files_rows),max(files_cols),min(files_cols), float_count, int_count,object_count

def benchmarks_stats(directories,benchmark_names):
    all_count_numeric = []
    all_count_non_numeric = []
    all_null = []
    all_ttl_rows = []
    all_ttl_cols = []
    max_rows = []
    min_rows = []
    max_cols = []
    min_cols = []
    avg_rows = []
    avg_cols = []
    ttl_files =[]
    all_float_col =[]
    all_int_col=[]
    all_object_col=[]
    for directory in tqdm(directories):
        files_count,count_numeric_cols, count_non_numeric_cols,nulls, ttl_rows, ttl_cols, max_row, min_row, max_col, min_col,float_cols, int_cols,object_cols = get_stats(None,directory)
        all_count_numeric.append(count_numeric_cols)
        all_count_non_numeric.append(count_non_numeric_cols)
        all_null.append(nulls)
        all_ttl_rows.append(ttl_rows)
        all_ttl_cols.append(ttl_cols)
        max_rows.append(max_row)
        min_rows.append(min_row)
        max_cols.append(max_col)
        min_cols.append(min_col)
        avg_rows.append((ttl_rows/files_count))
        avg_cols.append((ttl_cols/files_count))
        ttl_files.append(files_count)
        all_float_col.append(float_cols)
        all_int_col.append(int_cols)
        all_object_col.append(object_cols)


    data = {'benchmarks':benchmark_names,'path':directories,'ttl_files':ttl_files,'ttl_rows':all_ttl_rows,'ttl_cols':all_ttl_cols
            ,'max_row':max_rows,'min_row':min_rows,'max_col':max_cols,'min_col':min_cols,'avg_row':avg_rows,'avg_col':avg_cols,'nulls':all_null,'cols_non_numeric':all_count_non_numeric,
            'cols_numeric':all_count_numeric,'float64_cols':all_float_col,'int64_cols':all_int_col,'object_cols':all_object_col}

    summary_df = pd.DataFrame(data)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='benchmark stats.')
    parser.add_argument('stat_type', choices=['union_benchmarks', 'join_benchmarks','query_columns','query_table'], help='Type of statistic to collect')
    args = parser.parse_args()
    stat_type = args.stat_type
    res_path = f'benchmark_stats/{stat_type}' ## path to save the results
    os.makedirs(res_path,exist_ok=True)
    time = datetime.timestamp(datetime.now())
    file_name = f'{stat_type}_{time}.csv'
    result_csv = os.path.join(res_path,file_name)
    print(result_csv)
    print(stat_type)
    if 'benchmarks' in stat_type:
        if stat_type=='union_benchmarks':
            ### place the paths for the directories here
            directories = ['pylon/data/pylon_benchmark/source','starmie/data/table-union-search-benchmark/small/benchmark','starmie/data/table-union-search-benchmark/large/benchmark','starmie/data/santos/datalake']
            benchmark_names = ['pylon','tus_small','tus_large','santos']
        elif stat_type=='join_benchmarks':
            ### place the paths for the directories here
            directories=['nextiajd/testbedS/datasets','nextiajd/testbedM/datasets','lakebench/datalake/webtable/data_ssd/webtable/small_var1','lakebench/datalake/webtable/data_ssd/webtable/small_var2']
            benchmark_names=['nextiajd_small','nextiajd_medium','webtable_sampled_var1','webtable_sampled_var2']
        summary_df = benchmarks_stats(directories,benchmark_names)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='benchmarks', y='ttl_files')
        plt.title('Counts of File Count Across Benchmarks')
        plt.xlabel('Benchmarks')
        plt.ylabel('Total Files')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(res_path,'barplot_counts_of_file_count_across_benchmarks.png'))
        plt.close()
        # Bar Plot of Counts of column Count Across Benchmarks
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='benchmarks', y='ttl_cols')
        plt.title('Counts of File Count Across Benchmarks')
        plt.xlabel('Benchmarks')
        plt.ylabel('Total Files')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(res_path,'barplot_counts_of_columns_count_across_benchmarks.png'))
        plt.close()
        ####
    elif 'query' in stat_type:
        if stat_type=='query_columns':
            ### place the path of the dictionaries and the directories here
            # dictionaries = ['nextiajd/testbedS/join_dict_testbedS_warpgate.pkl','nextiajd/testbedS/join_dict_testbedS_warpgate_non-numerical_test.pkl','nextiajd/testbedS/join_dict_testbedS_10_non-numerical.pkl','nextiajd/testbedM/join_dict_testbedM_wrapgate.pkl','lakebench/join_dict_webtable_small.pkl','lakebench/join_dict_webtables_small_var2.pkl']
            # directories = ['nextiajd/testbedS/datasets','nextiajd/testbedS/datasets','nextiajd/testbedS/datasets','nextiajd/testbedM/datasets','lakebench/datalake/webtable/data_ssd/webtable/small_var1','lakebench/datalake/webtable/data_ssd/webtable/small_var2']
            # benchmark_names=['nextiajd_small','nextiajd_small_var2','nextiajd_small_var3','nextiajd_medium','webtable_small_var1','webtable_small_var2']
            dictionaries = ['/home/almutawa/inference/inference/lakebench/webtable/join_dict_final.pkl']
            directories = ['/home/almutawa/lakebench/webtable/data_ssd/webtable/large/split_1']
            benchmark_names=['lakebench']
            summary_df = query_columns_stats(benchmark_names,dictionaries,directories)
        elif stat_type =='query_table':
            ### place the path of the dictionaries and the directories here
            dictionaries = ['starmie/data/santos/santosUnionBenchmark.pickle','starmie/data/table-union-search-benchmark/small/sampled/tusLabeledtusUnionBenchmark','starmie/data/table-union-search-benchmark/large/sampled/tusLabeledtusLargeUnionBenchmark','pylon/data/pylon_benchmark/all_ground_truth_sans_recall.pkl']
            directories = ['starmie/data/santos/datalake','starmie/data/table-union-search-benchmark/small/benchmark','starmie/data/table-union-search-benchmark/large/benchmark','pylon/data/pylon_benchmark/source']
            benchmark_names=['santos','tus_small','tus_large','pylon']
            summary_df = query_tables_stats(benchmark_names,dictionaries,directories)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='benchmarks', y='ttl_queries')
        plt.title('Counts of Total Queries Across Benchmarks')
        plt.xlabel('Benchmarks')
        plt.ylabel('Total Queries')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(res_path,'barplot_counts_of_queries_across_benchmarks.png'))
        plt.close()
        # Bar Plot of Counts of column Count Across Benchmarks
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='benchmarks', y='nulls')
        plt.title('Counts of Nulls in Query Across Benchmarks')
        plt.xlabel('Benchmarks')
        plt.ylabel('Total Files')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(res_path,'barplot_counts_of_nulls_benchmarks.png'))
        plt.close()
        ####
    summary_df.to_csv(result_csv,index=False)   

if __name__ == "__main__":
    main()