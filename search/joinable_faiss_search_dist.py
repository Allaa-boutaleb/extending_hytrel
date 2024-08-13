import pickle
import sys
import os 
import faiss
import numpy as np
import time
import memory_profiler
import configs.search_configs as search_configs
import itertools
import logging
import sys
import traceback

def build_index(vectors):
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # Ensure dimension matches the aggregation method
    index.add(vectors)
    return index

def build_index_pq(vectors):
    num_cent_ids = 8
    nlist = 200
    cent_bits = 8
    vec_dim = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(vec_dim)
    index = faiss.IndexIVFPQ(quantizer, vec_dim, nlist, num_cent_ids, cent_bits)
    print('='*10, 'Training index', '='*10)
    index.train(vectors)
    print('='*10, 'Adding vectors', '='*10)
    index.add(vectors)
    return index

def joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k):
    num_datalake_cols= len(datalake_columns_hytrel)
    print('num_datalake_columns: ', num_datalake_cols)
    dataset_col = []
    vectors = []
    for pair, tensor in datalake_columns_hytrel:
        vectors.append(tensor)
        dataset_col.append(pair) 
    start_build = time.time()
    vectors = np.array(vectors)
    if search_configs.input['method'] == 'faiss_quantizer':
        print('build index using faiss quantizer')
        index = build_index_pq(vectors)
    else: 
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
                res[pair] = [(dataset_col[i],distances[0][i]) for i in indices[0]]
            elif benchmark in ('testbedS', 'testbedM'): 
                res[pair] = [dataset_col[i] for i in indices[0] if pair != dataset_col[i] and pair[0] != dataset_col[i][0]][:10]
    return res, build_duration, query_duration

# def track_memory_usage(): ## uncomment if you want to track memory usage
#     global benchmark 
#     benchmark = 'lakebench'
#     run_id = 0
#     size = 'small_var2'
#     with open(f'inference/inference/lakebench/webtables/{size}/vectors/hytrel_query_columns_{run_id}.pkl', 'rb') as f:
#          query_columns_hytrel = pickle.load(f)

#     with open(f'inference/inference/lakebench/webtables/{size}/vectors/hytrel_datasets_columns_{run_id}.pkl', 'rb') as f:
#         datalake_columns_hytrel = pickle.load(f)
    
#     k = 1000
#     res, build_duration, query_duration = joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k)
#     return res, build_duration, query_duration

def main():
    # run_id = [0,1]
    # size = 'small'
    # variation = 'small'
    # metric = 'cosine'
    # encoder = 'hytrel'
    # global benchmark
    # benchmark = 'lakebench'
    query_columns = search_configs.input['embedding_query_source']
    global benchmark
    benchmark = search_configs.input['datalake']
    k = search_configs.k[benchmark]
    with open(query_columns, 'rb') as f:
        query_columns_hytrel = pickle.load(f)
    print('============ loading vectors from multiple directories ============ \n')
    datalake_columns_hytrel = []
    for subfolder in search_configs.multiple_vector_dir['index']:
        datalake_columns = os.path.join(search_configs.input['embedding_source'], subfolder, search_configs.multiple_vector_dir['subfolder'],search_configs.multiple_vector_dir['file_name'])
        datalake_columns_hytrel.append(datalake_columns)
        with open(datalake_columns, 'rb') as f:
            datalake_columns_hytrel.extend(pickle.load(f))
        print('============ loading vectors done ============ \n')


        print(f'using query vectors from {query_columns}\n')
        print(f'using datalake vectors from {datalake_columns}\n')
        print(f'============ search started ============\n')
        if benchmark == 'lakebench':
            res, build_duration, query_duration = joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k)
        else:
            retieve_more_than_k = 1000 ##just for nextiajd 
            res, build_duration, query_duration = joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,retieve_more_than_k)
        print(f'============ search end ============\n')
        print(f'index build time: {build_duration} seconds')
        print(f'index build time: {query_duration} seconds')
        print('ttl time: ', build_duration + query_duration)
        res_dir = search_configs.output['path']
        os.makedirs(res_dir, exist_ok=True)
        candidates_pkl = os.path.join(res_dir,subfolder,search_configs.output['candidates'])
        with open(candidates_pkl, 'wb') as f:
            pickle.dump(res, f)

        print(f'candidates saved in: {candidates_pkl}\n')    

if __name__ == '__main__':
    main()
    # mem_usage = memory_profiler.memory_usage(track_memory_usage)
    # print(f'memory usage: {mem_usage}')
    # print(f'max memory usage: {max(mem_usage)}')