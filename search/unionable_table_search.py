import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from loguru import logger

# Clustering-based search functions

def get_column_clusters(table_name: str, df_clustering: pd.DataFrame) -> np.ndarray:
    """
    Get the unique cluster assignments for columns of a given table.
    
    Args:
        table_name (str): Name of the table.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
    
    Returns:
        np.ndarray: Array of unique cluster assignments for the table's columns.
    """
    return df_clustering[df_clustering['dataset'] == table_name]['cluster'].unique()

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
    Calculate the Jaccard similarity score between two tables based on their column clusters.
    
    Args:
        table1 (str): Name of the first table.
        table2 (str): Name of the second table.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
    
    Returns:
        float: Jaccard similarity score.
    """
    clusters_t1 = set(get_column_clusters(table1, df_clustering))
    clusters_t2 = set(get_column_clusters(table2, df_clustering))
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
    Perform unionable table search using clustering-based method for multiple queries.
    
    Args:
        queries (List[str]): List of query table names.
        datalake (List[str]): List of all table names in the data lake.
        df_clustering (pd.DataFrame): DataFrame containing clustering results.
        k (int): Number of top results to return for each query.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping each query table to its top-k similar tables.
    """
    logger.info(f"Starting unionable table search using clustering for {len(queries)} queries")
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