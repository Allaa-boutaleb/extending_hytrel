#!/usr/bin/env python

input = {
    "datalake": "santos",
    "datalake_source": '/home/almutawa/starmie/data/santos/datalake',
    "embedding_source" : '/home/almutawa/inference/inference/santos/vectors/hytrel_datalake_columns_0.pkl',
    "embedding_query_source" : '/home/almutawa/inference/inference/santos/vectors/hytrel_query_columns_0.pkl',
    "downstream_task": "union", ## this dictates the format of the embeddings saved 
    "method": 'clustering_based'
}
clustering = {
    "cluster_assignment": '/home/almutawa/inference/inference/santos/clustering/clustering_811_santos_run_id_0.pkl', ##place the path of the cluster assignment results  
}

union_faiss = {
    'compress_method': 'max'
}

k = {
    'santos': 10,
    'tus': 60,
    'tusLarge': 60,
    'pylon': 10
}

output = {
    'path': '/home/almutawa/inference/search/santos/faiss',
    'candidates': 'candidates_max_0.pkl'
}