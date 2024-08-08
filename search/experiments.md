## Experiments
After computing embeddings based on the desired configurations as explained [here](embedding_computation/experiments.md), you can run the desired search algorithm. <br>
Specify the downstream_task in [search configuration](configs/search_configs.py) as either ```join``` or ```union```
### Unionable table search 

#### Clustering-based 
Once you have executed clustering algorithm as described [here](clustering/experiments.md), you can follow the below instructions to preform the search. 

````python
input = {
    "datalake": "[name of the data lake]",
    "datalake_source": '[path to data lake]',
    "embedding_source" : '[path to data lake vectors]',
    "embedding_query_source" : '[path to the query vectors]',
    "downstream_task": "union", 
    "method": 'clustering'
}
clustering = {
    "cluster_assignment": '[path to clustering assignment obtained]' 
}
....
````

#### Flat index

compression_method can be one of the following: 1. max, 2. mean, 3. sum 

```python 
input = {
    "datalake": "[name of the data lake]",
    "datalake_source": '[path to data lake]',
    "embedding_source" : '[path to data lake vectors]',
    "embedding_query_source" : '[path to the query vectors]',
    "downstream_task": "union", 
    "method": 'flat-index'
}
....

union_faiss = {
    'compress_method': '[insert the compression method desired]'
}
....
```
### Joinable table search 

#### Flat index

```python 
input = {
    "datalake": "[name of the data lake]",
    "datalake_source": '[path to data lake]',
    "embedding_source" : '[path to data lake vectors]',
    "embedding_query_source" : '[path to the query vectors]',
    "downstream_task": "join", 
    "method": 'flat-index'
}
....
```



