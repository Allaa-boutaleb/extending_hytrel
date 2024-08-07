## Clustering module 
Module to run hierarchal clustering algorithm on the generated column embedding. 

### Steps:
1. Adjust parameters in configs.column_clustering.py
2. Execute column_embedding_clustering.py

#### Parameters: 
- input: embedding_source, datalake_source, 
- output: result, file_name'
- clustering: n_clusters_available

````
#!/usr/bin/env python
input = {
    "datalake": ### data lake name (i.e. santos)
    "datalake_source": ### path to the repository (i.e. where csv files reside)
    "embedding_source" : ### path to the pkl file where with the column embeddings
    "downstream_task": "union", ### always union 
}
clustering = {
    "n_clusters_available": False, ## true if you have a decided on number of cluster, otherwise, full clustering will be done 
    "experimental_thresholds":[0.8,0.6,0.4,0.2,0.195,0.19, 0.18,0.1], ## used if n_clusters_available is False
    "n_clusters": ### empty if n_clusters_available is False. otherwise, integer value specifying the number of clusters. 
    "affinity":"cosine",
    "linkage":'average'
}

output = {
    'result': ### path to save the results 
    'file_name': ### clustering result pkl file name 
}
````