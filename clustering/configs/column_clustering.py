#!/usr/bin/env python
from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATALAKE = "ugen_v2" # "santos", "ugen_v1", "ugen_v2" 

input = {
    "datalake": DATALAKE,
    "datalake_source": str(BASE_DIR / "data" / DATALAKE / "datalake"),
    "embedding_source": str(BASE_DIR / "inference" / DATALAKE / "vectors" / "hytrel_datalake_columns_0.pkl"),
    "downstream_task": "union",
}

clustering = {
    "method": "hdbscan",  # Options: "hierarchical" or "hdbscan"
    "n_clusters_available": True,
    "experimental_thresholds": [0.8, 0.6, 0.4, 0.2, 0.195, 0.19, 0.18, 0.1],
    "n_clusters": 811,
    "hierarchical_metric": "cosine",
    "linkage": 'average',
    # HDBSCAN specific parameters
    "min_cluster_size": 5,
    "min_samples": 5,
    "hdbscan_metric": "cosine",

}

output = {
    'result': str(BASE_DIR / "inference" / DATALAKE / "clustering"),
}
