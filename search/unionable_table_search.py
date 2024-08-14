import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import faiss
import numpy as np
import time 
import os
import numpy as np
import pandas as pd
import time

#### clustering based
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


#### Faiss 

def approximate_unionable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k,compress_method='max'):
    compressed_query_vectors = []
    query_names = []
    for dataset,tensors in query_columns_hytrel:
        if compress_method == 'mean':
            compressed_query_vectors.append((dataset,np.mean(tensors,axis=0)))
        elif compress_method == 'sum':
            compressed_query_vectors.append((dataset,np.sum(tensors,axis=0)))
        elif compress_method == 'max':
            compressed_query_vectors.append((dataset,np.max(tensors,axis=0)))
        query_names.append(dataset) 
    compressed_vectors = []
    dataset_names = []
    for dataset, tensors in datalake_columns_hytrel:
        if compress_method == 'mean':
         compressed_vectors.append(np.mean(tensors,axis=0))
        elif compress_method == 'sum':
            compressed_vectors.append(np.sum(tensors,axis=0))
        elif compress_method == 'max':
            compressed_vectors.append(np.max(tensors,axis=0))
        dataset_names.append(dataset) 

    start_build = time.time()
    vectors = np.array(compressed_vectors)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # Ensure dimension matches the aggregation method
    index.add(vectors)
    end_build = time.time()
    build_duration = end_build - start_build
    res = {}
    query_duration = 0
    for query, query_vec in compressed_query_vectors:
        query_vector = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        start_q = time.time()
        distances, indices = index.search(query_vector, k)
        end_q = time.time()
        query_duration += end_q - start_q
        if query not in res:
            res[query] = [dataset_names[i] for i in indices[0]]
    return res, build_duration, query_duration