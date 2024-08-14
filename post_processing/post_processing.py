
import pickle
from datasketch import MinHashLSHEnsemble, MinHash
import configs.common as common_configs
import os
import filtering_reranking as fr

# def determine_delimiter(filepath):
#      # Determine delimiter based on file content
#         with open(filepath, 'r') as file:
#             first_line = file.readline()
#             if ';' in first_line:
#                 delimiter = ';'
#             elif '\t' in first_line:
#                 delimiter = '\t'
#             else:
#                 delimiter = ','
#         return delimiter

# def get_df(file_path,delimiter=None):
#     delimiter = determine_delimiter(file_path)
#     print('file_path:',file_path)
#     try:
#         # if 'sewer-manholes.csv' in file_path or 'water-distribution-mains.csv' in file_path or 'district-wise-rainfall-data-for-india-2014.csv' in file_path or 'water-control-valves.csv' in file_path
#         if delimiter == ';':
#             df = pd.read_csv(file_path, delimiter=';')
#         else:
#             df = pd.read_csv(file_path)
#     except pd.errors.ParserError as e:
#         delimiter = determine_delimiter(file_path)
#         print(f'delimiter: {delimiter}')
#         df = pd.read_csv(file_path,delimiter=delimiter, encoding='utf-8',on_bad_lines='skip',low_memory=False)
#     except UnicodeDecodeError:
#         try:
#             print(f"UnicodeDecodeError: Failed to read {file_path} with encoding. Trying a different encoding.")
#             df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1')
#         except pd.errors.ParserError as e:
#             df = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='skip')
    
#     return df 

# def exact_containment(q,t):
#     return float(len(q.intersection(t)))/float(len(q))

# def lsh_ensemble_all_tables(datalake_dir, candidate_tables, query_table, query_column,num_perm=128, threshold=0.8, num_part=32):
#     q_df = get_df(os.path.join(datalake_dir,query_table))
#     set1 = set(q_df[query_column]) ## minhash for the query table
#     m1 = MinHash(num_perm=num_perm)
#     for d in set1:
#         if isinstance(d, str):
#             m1.update(d.encode('utf8'))
#     lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
#     num_part=num_part)

#     list_indices = []
#     c = 0 
#     est_dict = {}
#     for table in candidate_tables:
#         c = c +1 
#         try:
#             c_df = get_df(os.path.join(datalake_dir,table[0]))
#             set2 = set(c_df[table[1]])
#         except Exception as e:
#             c_df = get_df(os.path.join(datalake_dir,table),delimiter=';')
#             set2 = set(c_df[table[1]])
#         m2 = MinHash(num_perm=num_perm)
#         for d in set2:
#             if isinstance(d, str):
#                 m2.update(d.encode('utf8'))
#         list_indices.append((table, m2, len(set2)))
#         # annot_dict[table] = exact_containment(set1, set2)
#         est_dict[table] = est_containment(m1, m2)

#     if len(list_indices) == 0:
#         return [],est_dict
#     else:
#         lshensemble.index(list_indices)
#     res = []
#     for key in lshensemble.query(m1, len(set1)):
#         res.append(key)
#     if res == None:
#         return []
#     return res,est_dict

# def est_containment(q,t):
#     jaccard = q.jaccard(t)
#     v = MinHash.union(q,t).count()
#     v = v / q.count()
#     est_containment = jaccard * v
#     return est_containment

# def run_lsh_ensemble(datalake_dir,candidate_tables, num_perm=128, threshold=0.8, num_part=32):
#     all_res = {}
#     all_est = {}
#     for query in candidate_tables:
#         print(f"Running LSH ensemble for query {query}")
#         query_table = query[0]
#         query_column = query[1]
#         res,est= lsh_ensemble_all_tables(datalake_dir, candidate_tables.get(query), query_table, query_column, num_perm=num_perm, threshold=threshold, num_part=num_part)
#         all_res[query] = res
#         all_est[query] = est

#     return all_res, all_est

# def rank_table(est_dict):
#     sorted_dict = {k: [item for item in dict(sorted(v.items(), key=lambda item: item[1], reverse=True))] for k, v in est_dict.items()}
#     return sorted_dict

def main(): 
    ### load results from candidate search on benchmark queries (computed by joinability_faiss_search.py)
    candidate_tables = common_configs.input['original_rank']
    source_datalake = common_configs.input['datalake_source']
    num_prem = common_configs.lshensemble_configs['num_perm']
    threshold = common_configs.lshensemble_configs['threshold']
    num_part = common_configs.lshensemble_configs['num_part']
    output = common_configs.output['path']
    filter_result = common_configs.output['filter_result']
    rerank_result = common_configs.output['rerank_result']
    os.makedirs(output, exist_ok=True)
    filter = os.path.join(output, filter_result)
    rerank = os.path.join(output, rerank_result)
    with open(candidate_tables, 'rb') as f:
        candidate_tables = pickle.load(f)
    print('===================== postprocessing start =====================')
    res,est = fr.run_lsh_ensemble(source_datalake, candidate_tables, num_perm=num_prem, threshold=threshold, num_part=num_part)
    print('===================== postprocessing end =====================')
    print('===================== saving filtering result =====================')
    with open(filter, 'wb') as f:
        pickle.dump(res, f)
    print(f'saved in {filter}')
    print('===================== saving reranking result =====================')
    sorted_dict = fr.rank_table(est)
    with open(rerank, 'wb') as f:
        pickle.dump(sorted_dict, f)
    print(f'saved in {rerank}')

if __name__ == '__main__':
    main()

