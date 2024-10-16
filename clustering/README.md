## Clustering module 
Module to run hierarchical clustering or HDBSCAN algorithm on the generated column embeddings. 

### Steps:
1. Adjust parameters in configs.column_clustering.py
2. Execute column_embedding_clustering.py

#### Parameters: 
- input: embedding_source, datalake_source, datalake
- output: result
- clustering: method, n_clusters_available, experimental_thresholds, n_clusters, hierarchical_metric, hdbscan_metric, linkage, min_cluster_size, min_samples, cluster_selection_epsilon

```python
#!/usr/bin/env python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

input = {
    "datalake": "santos",  # data lake name (e.g., santos)
    "datalake_source": str(BASE_DIR / "data" / "santos" / "datalake"),  # path to the repository (where csv files reside)
    "embedding_source": str(BASE_DIR / "inference" / "santos" / "vectors" / "hytrel_datalake_columns_0.pkl"),  # path to the pkl file with the column embeddings
    "downstream_task": "union",  # always union 
}

clustering = {
    "method": "hierarchical",  # Options: "hierarchical" or "hdbscan"
    "n_clusters_available": False,  # True if you have decided on a number of clusters, otherwise full clustering will be done 
    "experimental_thresholds": [0.8, 0.6, 0.4, 0.2, 0.195, 0.19, 0.18, 0.1],  # Used if n_clusters_available is False
    "n_clusters": None,  # Used if n_clusters_available is True
    "hierarchical_metric": "cosine",
    "hdbscan_metric": "euclidean",
    "linkage": 'average',
    # HDBSCAN specific parameters
    "min_cluster_size": 5,
    "min_samples": 5,
    "cluster_selection_epsilon": 0.1
}

output = {
    'result': str(BASE_DIR / "inference" / "santos" / "clustering"),  # path to save the results 
}