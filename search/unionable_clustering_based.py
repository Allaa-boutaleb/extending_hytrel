import os
import numpy as np
import pandas as pd
import pickle
from utils.utils import *
import configs.search_configs as search_configs
import time

def get_column_clusters(table_name,df_clustering): 
    return df_clustering[df_clustering['dataset']==table_name]['cluster'].unique()

def get_intersect(set1,set2):
    intersection = list(set1 & set2)
    return intersection

def get_difference(set1, set2):
    diff = set1 ^ set2
    return diff

def get_union(set1, set2):
    union = set1.union(set2)
    return union

def get_score(table1, table2,df_clustering):
    clusters_t1 = set(get_column_clusters(table1,df_clustering))
    clusters_t2 = set(get_column_clusters(table2,df_clustering))
    intersect = get_intersect(clusters_t1,clusters_t2)
    union = get_union(clusters_t1,clusters_t2)
    jaccard_index = len(intersect) / len(union) if len(union) != 0 else 0
    
    return jaccard_index

def get_top_k(query_table, datalake, df_clustering, k=10):
    scores = {}
    for table in datalake:
        scores[table] = get_score(query_table, table,df_clustering)  # Assuming this function call is correct
    
    sorted_scores = sorted(scores.items(), key=lambda item: item[1],reverse=True)
    top_k = []
    for dataset, distance in sorted_scores[:k]:
        top_k.append(dataset)

    return top_k

def unionable_table_search_using_clustering(queries,datalake,df_clustering,k=10):
    res = {}
    for query_table in queries:
        res[query_table] = get_top_k(query_table, datalake, df_clustering,k)
    return res
def main():
    np.random.seed(42)
    res_dir = search_configs.output['path']
    os.makedirs(res_dir, exist_ok=True)
    candidates_pkl = os.path.join(res_dir,search_configs.output['candidates'])
    clustering = search_configs.clustering['cluster_assignment']
    query_file = search_configs.input['embedding_query_source']
    with open(query_file, 'rb') as f:
        query_columns_hytrel = pickle.load(f)
    query = []
    for query_table, tensors in query_columns_hytrel:
        query_table = query_table
        query.append(query_table)

    with open(clustering, 'rb') as f:
        clustering_res = pickle.load(f)
        print(f'using clustering results from {clustering}\n')
        print(f'using query vectors from {query_file}\n')
        print(f"number of clusters: {len(set(clustering_res['cluster'].unique()))} \n")   
        datalake = list(set(clustering_res['dataset'].unique()))
        print('=========== search using clustering started ===========')
        start_time = time.time()
        res = unionable_table_search_using_clustering(query, datalake, clustering_res,10)
        end_time = time.time()
        print('=========== search using clustering ended ===========')
        print(f'execution time: {end_time - start_time} seconds')
        with open(candidates_pkl, 'wb') as f:
            pickle.dump(res, f)
        print(f'candidates saved in: {candidates_pkl}\n')
if __name__ == '__main__':
    main()
