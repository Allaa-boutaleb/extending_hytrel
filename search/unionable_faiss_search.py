import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pickle 
import torch
import faiss
import numpy as np
import time 
import configs.search_configs as search_configs
import unionable_table_search as uts
from loguru import logger 

# def approximate_unionable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k,compress_method='max'):
#     compressed_query_vectors = []
#     query_names = []
#     for dataset,tensors in query_columns_hytrel:
#         if compress_method == 'mean':
#             compressed_query_vectors.append((dataset,np.mean(tensors,axis=0)))
#         elif compress_method == 'sum':
#             compressed_query_vectors.append((dataset,np.sum(tensors,axis=0)))
#         elif compress_method == 'max':
#             compressed_query_vectors.append((dataset,np.max(tensors,axis=0)))
#         query_names.append(dataset) 
#     compressed_vectors = []
#     dataset_names = []
#     for dataset, tensors in datalake_columns_hytrel:
#         if compress_method == 'mean':
#          compressed_vectors.append(np.mean(tensors,axis=0))
#         elif compress_method == 'sum':
#             compressed_vectors.append(np.sum(tensors,axis=0))
#         elif compress_method == 'max':
#             compressed_vectors.append(np.max(tensors,axis=0))
#         dataset_names.append(dataset) 

#     start_build = time.time()
#     vectors = np.array(compressed_vectors)
#     faiss.normalize_L2(vectors)
#     index = faiss.IndexFlatIP(vectors.shape[1])  # Ensure dimension matches the aggregation method
#     index.add(vectors)
#     end_build = time.time()
#     build_duration = end_build - start_build
#     res = {}
#     query_duration = 0
#     for query, query_vec in compressed_query_vectors:
#         query_vector = query_vec.reshape(1, -1)
#         faiss.normalize_L2(query_vector)
#         start_q = time.time()
#         distances, indices = index.search(query_vector, k)
#         end_q = time.time()
#         query_duration += end_q - start_q
#         if query not in res:
#             res[query] = [dataset_names[i] for i in indices[0]]
#     return res, build_duration, query_duration

"""
def main():
    np.random.seed(42)
    datalake = search_configs.input['datalake']
    k = search_configs.k[datalake]
    res_dir = search_configs.output['path']
    os.makedirs(res_dir, exist_ok=True)
    candidates_pkl = os.path.join(res_dir,search_configs.output['candidates'])
    datalake_columns = search_configs.input['embedding_source']
    query_columns = search_configs.input['embedding_query_source']
    compress_method = search_configs.union_faiss['compress_method']
    os.makedirs(res_dir, exist_ok=True)
    with open(query_columns, 'rb') as f:
        query_columns_hytrel = pickle.load(f)

    with open(datalake_columns, 'rb') as f:
        datalake_columns_hytrel = pickle.load(f)

    print(f'using {compress_method} to compress the vectors')
    res, build_duration, query_duration = uts.approximate_unionable_dataset_search(query_columns_hytrel, datalake_columns_hytrel,k,compress_method)
    with open(candidates_pkl, 'wb') as f:
        pickle.dump(res, f)
    print(f'execution time - index build : {build_duration} seconds')
    print(f'execution time - query : {build_duration} seconds')
    print(f'execution time - total : {build_duration + query_duration} seconds')
    print(f'candidates saved in: {candidates_pkl}\n')
"""

def main():
    np.random.seed(42)
    datalake = search_configs.input['datalake']
    k = search_configs.k[datalake]
    res_dir = search_configs.output['path']
    os.makedirs(res_dir, exist_ok=True)
    candidates_pkl = os.path.join(res_dir, search_configs.output['candidates'])
    
    # Load data
    with open(search_configs.input['embedding_query_source'], 'rb') as f:
        query_columns_hytrel = pickle.load(f)
    with open(search_configs.input['embedding_source'], 'rb') as f:
        datalake_columns_hytrel = pickle.load(f)

    start_time = time.time()
    
    if search_configs.union_faiss['use_two_step']:
        logger.info("Using two-step search with initial filtering")
        res, timing_stats = uts.unionable_table_search_faiss(
            query_columns_hytrel,
            datalake_columns_hytrel,
            k=k,
            use_two_step=True,
            initial_filter_k=search_configs.union_faiss['initial_filter_k'],
            compress_method=search_configs.union_faiss['compress_method']
        )
        logger.info(f"Two-step timing stats:")
        for key, val in timing_stats.items():
            logger.info(f"  {key}: {val:.2f}s")
    else:
        logger.info(f"Using single-step {'aggregation' if search_configs.union_faiss['compress_method'] else 'column-wise'} search")
        res, timing_stats = uts.unionable_table_search_faiss(
            query_columns_hytrel,
            datalake_columns_hytrel,
            k=k,
            use_two_step=False,
            compress_method=search_configs.union_faiss['compress_method']
        )
        logger.info(f"Single-step timing stats:")
        for key, val in timing_stats.items():
            logger.info(f"  {key}: {val:.2f}s")

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f}s")

    # Save results
    with open(candidates_pkl, 'wb') as f:
        pickle.dump(res, f)
    

if __name__ == '__main__':
    main()




