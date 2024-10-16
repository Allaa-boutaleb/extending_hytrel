## Experiments 
We run hierarchical clustering and HDBSCAN on embeddings produced with configurations 0 and 21. 
Details on the embedding computation configurations are [here](/embedding_computation/experiments.md). 

### Reproduce 
Place the following in the column clustering [configuration file](configs/column_clustering.py)

#### Hierarchical Clustering (Full clustering): 

```python 
clustering = {
    "method": "hierarchical",
    "n_clusters_available": False, 
    "experimental_thresholds": [0.8, 0.6, 0.4, 0.2, 0.195, 0.19, 0.18, 0.1], 
    "n_clusters": None,
    "hierarchical_metric": "cosine",
    "linkage": 'average'
}
```

#### Hierarchical Clustering (With specific cluster count): 
```python 
clustering = {
    "method": "hierarchical",
    "n_clusters_available": True, 
    "experimental_thresholds": None, 
    "n_clusters": [insert desired cluster count], 
    "hierarchical_metric": "cosine",
    "linkage": 'average'
}
```

#### HDBSCAN:
```python
clustering = {
    "method": "hdbscan",
    "hdbscan_metric": "euclidean",
    "min_cluster_size": 5,
    "min_samples": 5,
    "cluster_selection_epsilon": 0.1
}
```

### Running Experiments
1. Update the `configs/column_clustering.py` file with the desired configuration.
2. Run the `column_embedding_clustering.py` script.
3. Results will be saved in a new directory under the specified output path, with the naming convention: `run_{run_id}_{clustering_method}_{datalake_name}`.
4. Each run directory will contain the clustering results, configuration file, and relevant visualizations.
