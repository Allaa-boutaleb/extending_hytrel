import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from loguru import logger
import time

# Clustering-based search functions

def get_column_clusters(table_name: str, df_clustering: pd.DataFrame) -> np.ndarray:
    """
    Get the unique cluster assignments for columns of a given table, excluding the outlier cluster (-1).
    
    Args:
        table_name (str): Name of the table.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
    
    Returns:
        np.ndarray: Array of unique cluster assignments for the table's columns, excluding outliers.
    """
    # Explicitly filter out cluster -1 (outliers/noise)
    return df_clustering[(df_clustering['dataset'] == table_name) & 
                        (df_clustering['cluster'] != -1)]['cluster'].unique()


def get_intersect(set1: Set, set2: Set) -> List:
    """
    Get the intersection of two sets.
    
    Args:
        set1 (Set): First set.
        set2 (Set): Second set.
    
    Returns:
        List: List of elements in the intersection of set1 and set2.
    """
    return list(set1 & set2)

def get_union(set1: Set, set2: Set) -> Set:
    """
    Get the union of two sets.
    
    Args:
        set1 (Set): First set.
        set2 (Set): Second set.
    
    Returns:
        Set: Union of set1 and set2.
    """
    return set1.union(set2)

def get_score(table1: str, table2: str, df_clustering: pd.DataFrame) -> float:
    """
    Calculate the Jaccard similarity score between two tables based on their non-outlier column clusters.
    
    Args:
        table1 (str): Name of the first table.
        table2 (str): Name of the second table.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
    
    Returns:
        float: Jaccard similarity score based only on valid clusters.
    """
    # Get clusters excluding outliers
    clusters_t1 = set(get_column_clusters(table1, df_clustering))
    clusters_t2 = set(get_column_clusters(table2, df_clustering))
    
    # Handle cases where all columns might be outliers
    if len(clusters_t1) == 0 or len(clusters_t2) == 0:
        return 0.0
    
    intersect = get_intersect(clusters_t1, clusters_t2)
    union = get_union(clusters_t1, clusters_t2)
    jaccard_index = len(intersect) / len(union) if len(union) != 0 else 0
    
    return jaccard_index

def get_top_k(query_table: str, datalake: List[str], df_clustering: pd.DataFrame, k: int = 10) -> List[str]:
    """
    Get the top-k most similar tables to the query table based on Jaccard similarity of column clusters.
    
    Args:
        query_table (str): Name of the query table.
        datalake (List[str]): List of all table names in the data lake.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
        k (int): Number of top results to return.
    
    Returns:
        List[str]: List of top-k most similar table names.
    """
    scores = {table: get_score(query_table, table, df_clustering) for table in datalake}
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [dataset for dataset, _ in sorted_scores[:k]]

def unionable_table_search_using_clustering(queries: List[str], datalake: List[str], df_clustering: pd.DataFrame, k: int = 10) -> Dict[str, List[str]]:
    """
    Perform unionable table search using clustering-based method for multiple queries,
    considering only valid cluster assignments (excluding outliers).
    
    Args:
        queries (List[str]): List of query table names.
        datalake (List[str]): List of all table names in the data lake.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
        k (int): Number of top results to return for each query.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping each query table to its top-k similar tables.
    """
    logger.info(f"Starting unionable table search using clustering for {len(queries)} queries")
    
    # Log number of outlier columns for visibility
    total_cols = len(df_clustering)
    outlier_cols = len(df_clustering[df_clustering['cluster'] == -1])
    logger.info(f"Total columns: {total_cols}, Outlier columns: {outlier_cols} ({outlier_cols/total_cols*100:.2f}%)")
    
    res = {}
    for query_table in queries:
        res[query_table] = get_top_k(query_table, datalake, df_clustering, k)
        logger.debug(f"Completed search for query table: {query_table}")
    
    logger.success(f"Completed unionable table search using clustering")
    return res

# FAISS-based search functions

def approximate_unionable_dataset_search(
    query_columns_hytrel: List[Tuple[str, np.ndarray]],
    datalake_columns_hytrel: List[Tuple[str, np.ndarray]],
    k: int,
    compress_method: str = 'max'
) -> Tuple[Dict[str, List[str]], float, float]:
    """
    Perform approximate unionable dataset search using FAISS.
    
    Args:
        query_columns_hytrel (List[Tuple[str, np.ndarray]]): List of (table_name, column_embeddings) for query tables.
        datalake_columns_hytrel (List[Tuple[str, np.ndarray]]): List of (table_name, column_embeddings) for data lake tables.
        k (int): Number of top results to return for each query.
        compress_method (str): Method to compress column embeddings into a single vector ('mean', 'sum', or 'max').
    
    Returns:
        Tuple[Dict[str, List[str]], float, float]: 
            - Dictionary mapping each query table to its top-k similar tables.
            - Time taken to build the FAISS index.
            - Total time taken for all queries.
    """
    logger.info(f"Starting approximate unionable dataset search using FAISS with {compress_method} compression")
    
    compression_functions = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max
    }
    compress_func = compression_functions.get(compress_method)
    if not compress_func:
        logger.error(f"Invalid compression method: {compress_method}")
        raise ValueError(f"Invalid compression method: {compress_method}")

    compressed_query_vectors = [(dataset, compress_func(tensors, axis=0)) for dataset, tensors in query_columns_hytrel]
    compressed_vectors = [compress_func(tensors, axis=0) for _, tensors in datalake_columns_hytrel]
    dataset_names = [dataset for dataset, _ in datalake_columns_hytrel]

    start_build = time.time()
    vectors = np.array(compressed_vectors)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    build_duration = time.time() - start_build
    logger.debug(f"FAISS index built in {build_duration:.2f} seconds")

    res = {}
    query_duration = 0
    for query, query_vec in compressed_query_vectors:
        query_vector = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        start_q = time.time()
        distances, indices = index.search(query_vector, k)
        query_duration += time.time() - start_q
        res[query] = [dataset_names[i] for i in indices[0]]
        logger.debug(f"Completed search for query table: {query}")

    logger.success(f"Completed approximate unionable dataset search using FAISS")
    logger.info(f"Total query time: {query_duration:.2f} seconds")
    return res, build_duration, query_duration




# Added hybrid FAISS

def column_wise_similarity_search(
    query_columns_hytrel: List[Tuple[str, np.ndarray]],
    datalake_columns_hytrel: List[Tuple[str, np.ndarray]],
    k: int
) -> Tuple[Dict[str, List[str]], float]:
    """
    Pure column-by-column similarity search using FAISS.
    """
    logger.info("Starting pure column-wise similarity search")
    start_time = time.time()
    
    # Build index of all column vectors
    all_column_vectors = []
    column_metadata = []  # Store (table_name, column_index)
    
    for table_name, table_vectors in datalake_columns_hytrel:
        for col_idx, col_vector in enumerate(table_vectors):
            all_column_vectors.append(col_vector)
            column_metadata.append((table_name, col_idx))
    
    vectors = np.array(all_column_vectors)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    
    # Search for each query table
    results = {}
    for query_name, query_columns in query_columns_hytrel:
        table_scores = {}  # {table_name: max_similarity}
        
        for query_col in query_columns:
            query_vector = query_col.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            distances, indices = index.search(query_vector, k*2)
            
            for dist, idx in zip(distances[0], indices[0]):
                table_name, _ = column_metadata[idx]
                if table_name not in table_scores:
                    table_scores[table_name] = 0
                table_scores[table_name] = max(table_scores[table_name], dist)
        
        # Get top-k tables based on scores
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results[query_name] = [table for table, _ in top_tables]

    total_time = time.time() - start_time
    logger.success(f"Completed pure column-wise search in {total_time:.2f} seconds")
    return results, total_time

def unionable_table_search_faiss(
    query_columns_hytrel: List[Tuple[str, np.ndarray]],
    datalake_columns_hytrel: List[Tuple[str, np.ndarray]],
    k: int,
    use_two_step: bool = False,
    initial_filter_k: int = 100,
    compress_method: str = 'max'
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    Unified search function that can do either:
    1. Pure aggregation-based search
    2. Pure column-wise search
    3. Two-step efficient search (aggregation filtering followed by column-wise)
    """
    timing_stats = {}
    
    if not use_two_step:
        # Pure single-method search
        if compress_method is not None:
            # Pure aggregation-based search
            logger.info(f"Using pure aggregation-based search with {compress_method} compression")
            results, build_time, query_time = approximate_unionable_dataset_search(
                query_columns_hytrel,
                datalake_columns_hytrel,
                k=k,
                compress_method=compress_method
            )
            timing_stats = {
                'build_time': build_time,
                'query_time': query_time,
                'total_time': build_time + query_time
            }
        else:
            # Pure column-wise search
            logger.info("Using pure column-wise search")
            results, total_time = column_wise_similarity_search(
                query_columns_hytrel,
                datalake_columns_hytrel,
                k=k
            )
            timing_stats = {'total_time': total_time}
    
    else:
        # Two-step efficient search
        logger.info(f"Using two-step search with {compress_method} aggregation filtering")
        start_time = time.time()
        
        # Step 1: Fast aggregation filtering
        filter_start = time.time()
        agg_results, build_time, filter_time = approximate_unionable_dataset_search(
            query_columns_hytrel,
            datalake_columns_hytrel,
            k=initial_filter_k,
            compress_method=compress_method
        )
        filter_duration = time.time() - filter_start
        logger.info(f"Filtering step completed in {filter_duration:.2f} seconds")
        
        # Step 2: Column-wise matching on filtered candidates
        matching_start = time.time()
        final_results = {}
        
        for query_name in agg_results:
            # Get filtered candidates
            filtered_candidates = agg_results[query_name]
            filtered_datalake = [(name, emb) for name, emb in datalake_columns_hytrel 
                               if name in filtered_candidates]
            
            # Do column-wise matching on filtered subset
            query_data = [(query_name, next(emb for qn, emb in query_columns_hytrel 
                                          if qn == query_name))]
            col_results, _ = column_wise_similarity_search(query_data, filtered_datalake, k)
            final_results[query_name] = col_results[query_name]
        
        matching_duration = time.time() - matching_start
        total_time = time.time() - start_time
        
        results = final_results
        timing_stats = {
            'filter_time': filter_duration,
            'matching_time': matching_duration,
            'total_time': total_time
        }
        
        logger.info(f"Column matching step completed in {matching_duration:.2f} seconds")
        logger.success(f"Two-step search completed in {total_time:.2f} seconds")
    
    return results, timing_stats