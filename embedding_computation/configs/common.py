"""
Common configuration settings for the HyTrel embedding computation.

This module contains the configuration parameters used across the project,
including global parameters, input and output paths, and computation settings.
"""

from typing import Dict, Any

global_params: Dict[str, Any] = {
    "hytrel_model": "/home/almutawa/hypergraph-tabular-lm/checkpoints/contrast/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt",
    "downstream_task": "union",
    "run_id": 0
}

input_config: Dict[str, str] = {
    "source": "/home/almutawa/starmie/data/santos/query",
    "type": "query"
}

computation: Dict[str, Any] = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '/home/almutawa/hypergraph-tabular-lm/logs',
    "log_file_name": "logs_run_id_0.txt",
    "save_auxiliary": False,
    "handle_null_column_names": False
}

output: Dict[str, str] = {
    "vectors": "/home/almutawa/inference/inference/santos/vectors",
    "auxiliary": "/home/almutawa/inference/inference/santos/auxiliary"
}