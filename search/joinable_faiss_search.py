import pickle
import sys
import os 
import faiss
import numpy as np
import time
import memory_profiler
import configs.search_configs as search_configs
import sys
import joinable_table_search as jts

# def build_index(vectors):
#     faiss.normalize_L2(vectors)
#     index = faiss.IndexFlatIP(vectors.shape[1])  # Ensure dimension matches the aggregation method
#     index.add(vectors)
#     return index

# def joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k,benchmark='nextiajd'):
#     num_datalake_cols= len(datalake_columns_hytrel)
#     print('num_datalake_columns: ', num_datalake_cols)
#     dataset_col = []
#     vectors = []
#     for pair, tensor in datalake_columns_hytrel:
#         vectors.append(tensor)
#         dataset_col.append(pair) 
#     start_build = time.time()
#     vectors = np.array(vectors)
#     print('build regular index using faiss')
#     index = build_index(vectors)
#     index = build_index(vectors)
#     end_build = time.time()
#     build_duration = end_build - start_build
#     res = {}
#     query_duration = 0
#     print(f'number of queries: {len(query_columns_hytrel)}')
#     for pair, query_vec in query_columns_hytrel:
#         query_vector = query_vec.reshape(1, -1)
#         faiss.normalize_L2(query_vector)
#         start_q = time.time()
#         distances, indices = index.search(query_vector, k)
#         end_q = time.time()
#         query_duration += end_q - start_q
#         if pair not in res:
#             if benchmark == 'lakebench':
#                 res[pair] = [dataset_col[i] for i in indices[0]]
#             elif benchmark in ('testbedS', 'testbedM'): 
#                 res[pair] = [dataset_col[i] for i in indices[0] if pair != dataset_col[i] and pair[0] != dataset_col[i][0]][:10]
#     return res, build_duration, query_duration

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
    print('============ loading vectors from 1 direcory ============ \n')
    datalake_columns = search_configs.input['embedding_source']
    with open(datalake_columns, 'rb') as f:
        datalake_columns_hytrel = pickle.load(f)
    print('============ loading vectors done ============ \n')

    query_columns = search_configs.input['embedding_query_source']
    benchmark = search_configs.input['datalake']
    k = search_configs.k[benchmark]
    index_type = search_configs.input['method']
    res_dir = search_configs.output['path']
    os.makedirs(res_dir, exist_ok=True)
    candidates_pkl = os.path.join(res_dir,search_configs.output['candidates'])
    with open(query_columns, 'rb') as f:
        query_columns_hytrel = pickle.load(f)


    print(f'using query vectors from {query_columns}\n')
    print(f'using datalake vectors from {datalake_columns}\n')
    print(f'============ search started ============\n')
    if benchmark == 'lakebench':
        res, build_duration, query_duration = jts.joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k,benchmark,index_type)
    else:
        retieve_more_than_k = 1000 ##just for nextiajd 
        res, build_duration, query_duration = jts.joinable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,retieve_more_than_k,benchmark)
    print(f'============ search end ============\n')
    print(f'index build time: {build_duration} seconds')
    print(f'index build time: {query_duration} seconds')
    print('ttl time: ', build_duration + query_duration)
        
    with open(candidates_pkl, 'wb') as f:
        pickle.dump(res, f)

    print(f'candidates saved in: {candidates_pkl}\n')    

if __name__ == '__main__':
    main()
    # mem_usage = memory_profiler.memory_usage(track_memory_usage)
    # print(f'memory usage: {mem_usage}')
    # print(f'max memory usage: {max(mem_usage)}')