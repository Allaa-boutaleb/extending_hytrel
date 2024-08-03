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

output = {
    'path': '/home/almutawa/inference/search/santos/clustering',
    'candidates': 'candidate_clusters_811_run_id_0.pkl'
}