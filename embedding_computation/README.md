## Embedding Computation module 
Module to compute embeddings using HyTrel pre-trained model. We make use of pre-trained HyTrel model that is provided by the [original authors](https://github.com/awslabs/hypergraph-tabular-lm) to encode tabular data with the purpose of extending and evaluating HyTrel for data discovery tasks. 
### Steps:
1. Adjust configuration in configs.common.py
2. Execute compute_embedding.py 
#### Parameters: 
- input: source, type
- output: vectors 
- global_params: hytrel_model
- computation:table_process, nrows, logs, log_file_name, column_names

``````
#!/usr/bin/env python
global_params = {
    "hytrel_model" : ### path for the hytrel pretrained model 
    "downstream_task": ### type of task [union or join] this will dictate the format 
    "run_id": ### integer value representing a run id (useful for tracking experiments)
}
input = {
    "source": ### path to dataset source (tables, csv files)
    "type": ### type of the dataset. [query or datalake]
}
computation = {
    "table_process": ### type of table process (refer to paper for list)
    "column_names": ### handling of column names 
    "nrows": ### integer value number of rows to sample. default is set to 30 
    "pandas_sample": ### using pandas sampling [False, True]
    "pandas_rate_sample": ### using pandas sampling [False, True]
    "logs": ### path for the logs 
    "log_file_name": ### log file name [txt file]
    "save_auxiliary": ### boolean value specifying wheter to save auxiliary files: jsonl format file for each dataset, mappings, and full table embeddings (per table)
    "handle_null_column_names": ### boolean value specifying whether to place an empty string for null column names. 
}
output = {
    "vectors": ### path to save the embedding vectors 
    "auxiliary": ### path to save the auxiliary data generated (if save_auxiliary is set to True)
}
``````

#### References: 
Thanks to the work cited below
`````
@inproceedings{NEURIPS2023_66178bea,
 author = {Chen, Pei and Sarkar, Soumajyoti and Lausen, Leonard and Srinivasan, Balasubramaniam and Zha, Sheng and Huang, Ruihong and Karypis, George},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {32173--32193},
 publisher = {Curran Associates, Inc.},
 title = {HyTrel: Hypergraph-enhanced  Tabular Data Representation Learning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/66178beae8f12fcd48699de95acc1152-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
`````