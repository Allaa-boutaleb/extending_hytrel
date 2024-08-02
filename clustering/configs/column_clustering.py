#!/usr/bin/env python

input = {
    "datalake": "santos",
    "datalake_source": '/home/almutawa/starmie/data/santos/datalake',
    "embedding_source" : '/home/almutawa/inference/inference/santos/vectors/hytrel_datalake_columns_0.pkl',
    "downstream_task": "union", ## this dictates the format of the embeddings saved 
}
clustering = {
    "n_clusters_available": False, ## true if you have a decided on number of cluster, otherwise, full clustering will be done 
    "experimental_thresholds":[0.8,0.6,0.4,0.2,0.195,0.19, 0.18,0.1], ## used if n_clusters_available is False
    "n_clusters": 811, 
    "affinity":"cosine",
    "linkage":'average'
}

output = {
    'result': '/home/almutawa/inference/inference/santos/clustering',
    'file_name': 'clustering_811_santos_run_id_0.pkl'
}