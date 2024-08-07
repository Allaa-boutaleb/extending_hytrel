#!/usr/bin/env python

input = {
    "original_rank" : "/Users/alaaalmutawa/Documents/Thesis/hytrel/hypergraph-tabular-lm/inference/inference/nextiajd/testbedS/test/candidates_4.pkl",
    "datalake_source":"/Users/alaaalmutawa/Documents/Thesis/nextiajd/testbedS/datasets", 
}

output = {
    'path': '/Users/alaaalmutawa/Documents/Thesis/hytrel/hypergraph-tabular-lm/inference/inference/nextiajd/testbedS/test/postprocessing',
    'filter_result': 'filtered_4.pkl',
    'rerank_result': 'rerank_candidates_4.pkl'

}

lshensemble_configs = {
    'num_perm':256, 
    'threshold':0.5, 
    'num_part':32
}