## Post-processing step module 
Module to run filtering and reranking post processing step. It is only applicable for joinable table search task. 
### Steps: 
1. Adjust configs.common.py
2. Run post_processing.py

#### Parameter setting: 
- input: original_rank, datalake_source
- lshensemble_configs: num_perm, threshold, num_part
- output: path, filter_result, rerank_result
`````
#!/usr/bin/env python
input = {
    "original_rank" : ### path to the pkl file containing result from joinable table search 
    "datalake_source": ### path to the datalake repository 
}

output = {
    'path': ### path for output 
    'filter_result': ### filtering result file name. has to be pkl file. 
    'rerank_result': ### rerank result file name. has to be pkl file. 
}

lshensemble_configs = {
    'num_perm': ### number of premutations 
    'threshold': ### jaccard setcontainment threshold 
    'num_part': ### number of partitions 
}
`````


