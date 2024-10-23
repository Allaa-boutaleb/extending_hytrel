#!/usr/bin/env python
from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

input = {
    "datalake": "santos",
    "datalake_source": str(BASE_DIR / "data" / "santos" / "datalake"),
    "embedding_source": str(BASE_DIR / "inference" / "santos" / "vectors" / "hytrel_datalake_columns_0.pkl"),
    "downstream_task": "union",
}

clustering = {
    "method": "hdbscan",  # Options: "hierarchical" or "hdbscan"
    "n_clusters_available": True,
    "experimental_thresholds": [0.8, 0.6, 0.4, 0.2, 0.195, 0.19, 0.18, 0.1],
    "n_clusters": 749,
    "hierarchical_metric": "cosine",
    "linkage": 'average',
    # HDBSCAN specific parameters
    "min_cluster_size": 5,
    "min_samples": 2,
    "hdbscan_metric": "euclidean",

}

output = {
    'result': str(BASE_DIR / "inference" / "santos" / "clustering"),
}
