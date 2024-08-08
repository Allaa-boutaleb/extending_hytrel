## Experiments 
Post processing step is preformed on joinable table search result for NextiaJD testbedS benchmark with query set1, set2, and set3. 
<br>
The module can be configured [here](configs/common.py). 

### LSHEnsemble configuration: 
In our experiments, LSHEnsemble index is configured with the below: configurations 
```python
lshensemble_configs = { ## experiments configs 
    'num_perm':256, 
    'threshold':0.5, 
    'num_part':32
}
```

### Reranking and filtering: 
The script will preform both reranking and filtering to a given candidate table list: 
```python 
input = {
    "original_rank" : "[path to pickle file containing results obtained from the search module]",
    "datalake_source":"[path to the data lake repository]", 
}
```

