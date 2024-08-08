## Experiments 
We run heirarcal clustering on embeddings produced with the following configration 0 and 21. 
<br> Details on the configuration are [here](/embedding_computation/experiments.md). 

### Reproduce 
Place the following in column clustering [configuration file](configs/column_clustering.py)

#### Full clustering: 

```python 
clustering = {
    "n_clusters_available": False, 
    "experimental_thresholds":[0.8,0.6,0.4,0.2,0.195,0.19, 0.18,0.1], 
    "n_clusters": None, 
    "affinity":"cosine",
    "linkage":'average'
}
```

#### With specific cluster count: 
```python 
clustering = {
    "n_clusters_available": True, 
    "experimental_thresholds":None, 
    "n_clusters": [insert desired cluster count], 
    "affinity":"cosine",
    "linkage":'average'
}
```