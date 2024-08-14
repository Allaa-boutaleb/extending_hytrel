import faiss
import numpy as np
import time

def build_index(vectors):
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # Ensure dimension matches the aggregation method
    index.add(vectors)
    return index

def joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k,benchmark='nextiajd'):
    num_datalake_cols= len(datalake_columns_hytrel)
    print('num_datalake_columns: ', num_datalake_cols)
    dataset_col = []
    vectors = []
    for pair, tensor in datalake_columns_hytrel:
        vectors.append(tensor)
        dataset_col.append(pair) 
    start_build = time.time()
    vectors = np.array(vectors)
    print('build regular index using faiss')
    index = build_index(vectors)
    index = build_index(vectors)
    end_build = time.time()
    build_duration = end_build - start_build
    res = {}
    query_duration = 0
    print(f'number of queries: {len(query_columns_hytrel)}')
    for pair, query_vec in query_columns_hytrel:
        query_vector = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        start_q = time.time()
        distances, indices = index.search(query_vector, k)
        end_q = time.time()
        query_duration += end_q - start_q
        if pair not in res:
            if benchmark == 'lakebench':
                res[pair] = [dataset_col[i] for i in indices[0]]
            elif benchmark in ('testbedS', 'testbedM'): 
                res[pair] = [dataset_col[i] for i in indices[0] if pair != dataset_col[i] and pair[0] != dataset_col[i][0]][:10]
    return res, build_duration, query_duration