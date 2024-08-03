#!/usr/bin/env python

input = {
    "datalake": "testbedS",
    "datalake_source": '/home/almutawa/starmie/data/santos/datalake',
    "embedding_source" : '/Users/alaaalmutawa/Documents/Thesis/hytrel/hypergraph-tabular-lm/inference/inference/nextiajd/testbedS/vectors/warpgate/hytrel_datalake_columns_4.pkl',
    "embedding_query_source" : '/Users/alaaalmutawa/Documents/Thesis/hytrel/hypergraph-tabular-lm/inference/inference/nextiajd/testbedS/vectors/warpgate/hytrel_query_columns_4.pkl',
    "downstream_task": "join", ## this dictates the format of the embeddings saved 
    "method": 'faiss'
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
    'webtables': 20

}

output = {
    'path': '/Users/alaaalmutawa/Documents/Thesis/hytrel/hypergraph-tabular-lm/inference/inference/nextiajd/testbedS/test',
    'candidates': 'candidates_4.pkl'
}