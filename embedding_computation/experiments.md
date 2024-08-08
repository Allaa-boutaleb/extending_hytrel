
## Experiments

We compute embeddings with the following configurations: 

### Unionable table search 
#### A. Initial Evaluation: 
For the below benchmarks, we compute embeddings with the following configurations. 
Benchmark   |Task    |   
|:---------------:|:------------:|
SANTOS | Union |
Pylon | Union |
TUS | Union |

To do so, place the following computation configuration in ```configs.common.py```: 
```python
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
#### B. Table Sampling: 
We run table sampling experiments on SANTOS benchmark. <br>
Below are the enumrated experiments: 
<hr>

```python
### 1 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 2 
computation = {
    "table_process": 'sort_by_tfidf',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 3 
computation = {
    "table_process": 'sort_by_tfidf_dropna',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 4 
computation = {
    "table_process": 'value_based_sort_column',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 5
computation = {
    "table_process": 'sort_col_independent_tfidf_dropna',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 6
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 7
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 100,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 8
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 150,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 9
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 500,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 10
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 1000,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 11
computation = {
    "table_process": 'sort_col_independent_tfidf_dropna',
    "column_names": None,
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 12
computation = {
    "table_process": 'value_based_sort_column',
    "column_names": None,
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 13
computation = {
    "table_process": 'value_based_sort_column',
    "column_names": None,
    "nrows": 1000,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 14
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 1000,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 15
computation = {
    "table_process": 'sort_col_independent_tfidf_dropna',
    "column_names": None,
    "nrows": 1000,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 16
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 2000,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 17
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```


<hr>

```python
### 18
computation = {
    "table_process": 'sort_col_independent_tfidf_dropna',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 19
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 20
computation = {
    "table_process": 'sort_col_independent_tfidf_dropna',
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 21
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 22
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 100,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 23
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": True,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 24
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": True,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 25
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": True,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 26
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 50,
    "pandas_sample": True,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```

<hr>

```python
### 27
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 2000,
    "pandas_sample": True,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": True
}
```
### Joinable table search 
#### A. Intial Evaluation 
For the below benchmarks, we compute embeddings with the following configurations. 
Benchmark   |Task    |   
|:---------------:|:------------:|
NextiaJD testbedS | Join |
NextiaJD testbedM | Join |
Lakebench: Webtables 1st variation | Join |
Lakebench: Webtables 2nd variation | Join |

Place the following computation configuration in ```configs.common.py```: 

```python
### 0 
computation = {
    "table_process": None,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

##### NextiaJD 

```python
### 1 
computation = {
    "table_process": None,
    "column_names": 'make_headers_null',
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 2 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": 'make_headers_null',
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 3 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": 'make_headers_null',
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 4 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": 'make_headers_null',
    "nrows": 100,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

```python
### 5 
computation = {
    "table_process": None,
    "column_names": 'make_headers_null',
    "nrows": 50,
    "pandas_sample": True,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```

<hr>

##### Lakebench: Webtable 

```python
### 1 
computation = {
    "table_process": 'prioritize_non_null_rows,
    "column_names": None,
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

The below configuration was ran on 2nd variation only: 

```python
### 2 
computation = {
    "table_process": None,
    "column_names": 'make_headers_null',
    "nrows": 30,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 3 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": 'make_headers_null',
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 4 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 50,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 5 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 100,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
<hr>

```python
### 6 
computation = {
    "table_process": 'prioritize_non_null_rows',
    "column_names": None,
    "nrows": 1000,
    "pandas_sample": False,
    "pandas_rate_sample": False,
    "logs": '[place the desired logs location]',
    "log_file_name": '[place the name of the log file]',
    "save_auxiliary": False,
    "handle_null_column_names": False
}
```
