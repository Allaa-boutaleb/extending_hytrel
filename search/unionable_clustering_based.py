import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict
from loguru import logger
import time

import configs.search_configs as search_configs
import unionable_table_search as uts

def main():
    """
    Main function to perform unionable table search using clustering-based method.
    """
    np.random.seed(42)
    
    # Set up output directory and files
    res_dir = search_configs.output['path']
    os.makedirs(res_dir, exist_ok=True)
    candidates_pkl = os.path.join(res_dir, search_configs.output['candidates'])
    
    # Load clustering results
    clustering_file = search_configs.clustering['cluster_assignment']
    with open(clustering_file, 'rb') as f:
        clustering_res = pickle.load(f)
    
    logger.info(f'Using clustering results from {clustering_file}')
    logger.info(f"Number of clusters: {len(set(clustering_res['cluster'].unique()))}")
    
    # Load query embeddings
    query_file = search_configs.input['embedding_query_source']
    with open(query_file, 'rb') as f:
        query_columns_hytrel = pickle.load(f)
    
    logger.info(f'Using query vectors from {query_file}')
    
    # Extract query table names
    query = [query_table for query_table, _ in query_columns_hytrel]
    
    # Get unique table names in the data lake
    datalake = list(set(clustering_res['dataset'].unique()))
    
    # Perform unionable table search
    logger.info('Starting search using clustering')
    start_time = time.time()
    res = uts.unionable_table_search_using_clustering(query, datalake, clustering_res, search_configs.k[search_configs.input['datalake']])
    end_time = time.time()
    logger.success('Search using clustering completed')
    logger.info(f'Execution time: {end_time - start_time:.2f} seconds')
    
    # Save results
    with open(candidates_pkl, 'wb') as f:
        pickle.dump(res, f)
    logger.info(f'Candidates saved in: {candidates_pkl}')

if __name__ == '__main__':
    logger.add("unionable_search.log", rotation="10 MB")
    main()