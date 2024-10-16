"""
Common configuration settings for the HyTrel embedding computation.

This module contains the configuration parameters used across the project,
including global parameters, input and output paths, and computation settings.
"""

import os
from typing import Dict, Any
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Function to get environment variable or default value
def get_env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)

global_params: Dict[str, Any] = {
    "hytrel_model": str(BASE_DIR / "checkpoints" / "contrast" / "epoch=4-step=32690.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt"),
    "downstream_task": get_env_or_default("DOWNSTREAM_TASK", "union"),
    "run_id": int(get_env_or_default("RUN_ID", "0"))
}

input: Dict[str, str] = {
    "source": str(BASE_DIR / "data" / "santos" / "datalake"),
    "type": "datalake"
}

computation: Dict[str, Any] = {
    "table_process": None,
    "column_names": None,
    "nrows": int(get_env_or_default("NROWS", "30")),
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": str(BASE_DIR / "logs"),
    "log_file_name": f"logs_run_id_{global_params['run_id']}.txt",
    "save_auxiliary": False,
    "handle_null_column_names": False
}

output: Dict[str, str] = {
    "vectors": str(BASE_DIR / "inference" / "santos" / "vectors"),
    "auxiliary": str(BASE_DIR / "inference" / "santos" / "auxiliary")
}