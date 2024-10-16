#!/usr/bin/env python
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

input = {
    "datalake": "santos",
    "datalake_source": str(BASE_DIR / "data" / "santos" / "datalake"),
    "embedding_source": str(BASE_DIR / "inference" / "santos" / "vectors" / "hytrel_datalake_columns_0.pkl"),
    "downstream_task": "union",  # This dictates the format of the saved embeddings
}

clustering = {
    "n_clusters_available": False,  # True if you have decided on a number of clusters, otherwise full clustering will be done
    "experimental_thresholds": [0.8, 0.6, 0.4, 0.2, 0.195, 0.19, 0.18, 0.1],  # Used if n_clusters_available is False
    "n_clusters": 811,  # Used if n_clusters_available is True
    "metric": "cosine",
    "linkage": 'average'
}

output = {
    'result': str(BASE_DIR / "inference" / "santos" / "clustering"),
    'file_name': 'clustering_811_santos_run_id_0.pkl'
}