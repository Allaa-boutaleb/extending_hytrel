#!/usr/bin/env python

input = {
    "datalake": "lakebench",
    "datalake_source": '/home/almutawa/lakebench/webtable',
    "embedding_source" : '/home/almutawa/inference/inference/lakebench/webtable',
    'embedding_source_distributed': True, 
    "embedding_query_source" : '/home/almutawa/inference/inference/lakebench/webtable/001/vectors/hytrel_query_columns_0.pkl',
    "downstream_task": "join", ## this dictates the format of the embeddings saved 
    "method": 'hnsw' ## faiss_hnsw or faiss_flat
}
multiple_vector_dir = { ## incase of distributed processing of the embeddings 
    'index':['001','002','003','004','005','006'],
    'subfolder': 'vectors',
    'file_name': 'hytrel_dataset_columns_0.pkl'
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
    'pylon': 10,
    'testbedS': 10,
    'testbedM': 10,
    'lakebench': 20

}

output = {
    'path': '/home/almutawa/inference/search/lakebench/webtable',
    'candidates': 'candidates_0.pkl'
}