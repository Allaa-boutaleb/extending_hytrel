#!/usr/bin/env python

global_params = {
    "hytrel_model" : "/home/almutawa/hypergraph-tabular-lm/checkpoints/contrast/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt",
    "downstream_task": "union",
    "run_id": 0
}
input = {
    "source":"/home/almutawa/starmie/data/santos/query",
    "type":"query"
}
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '/home/almutawa/hypergraph-tabular-lm/logs',
    "log_file_name": "loogs_run_id_0.txt",
    "save_auxiliary": False,
    "handle_null_column_names": False
}
output = {
    "vectors":"/home/almutawa/inference/inference/santos/vectors",
    "auxiliary": "/home/almutawa/inference/inference/santos/auxiliary"
}
    
