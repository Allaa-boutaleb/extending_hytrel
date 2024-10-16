import os
from typing import Dict, List, Tuple, Any
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
from loguru import logger

import configs.column_clustering as column_clustering_configs

def plot_dendrogram(model: AgglomerativeClustering, **kwargs: Any) -> np.ndarray:
    """
    Plot the dendrogram for the given AgglomerativeClustering model.

    Args:
        model (AgglomerativeClustering): The fitted clustering model.
        **kwargs: Additional keyword arguments for the dendrogram function.

    Returns:
        np.ndarray: The linkage matrix used for plotting the dendrogram.
    """
    # Count the number of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

def get_mapping(path: str, offset: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    Create a mapping DataFrame from the given path and offset information.

    Args:
        path (str): Path to the directory containing CSV files.
        offset (Dict[str, Dict[str, int]]): Offset information for each dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the mapping information.
    """
    columns = ['index', 'dataset', 'column_index', 'column_name', 'column_name_lower']
    df_map = pd.DataFrame(columns=columns)

    for i in offset:
        df_temp = pd.read_csv(f'{path}/{i}')
        columns = df_temp.columns.tolist()
        data = [
            {
                'index': offset[i]['index'],
                'dataset': i,
                'column_index': int(j),
                'column_name': columns[j],
                'column_name_lower': columns[j].lower()
            }
            for j in range(offset[i]['length'])
        ]
        df_map = pd.concat([df_map, pd.DataFrame(data)], ignore_index=True)
    
    return df_map

def get_offset_all_embeddings(embeddingfile: List[Tuple[str, np.ndarray]]) -> Tuple[Dict[str, Dict[str, int]], List[np.ndarray]]:
    """
    Extract offset information and all embeddings from the embedding file.

    Args:
        embeddingfile (List[Tuple[str, np.ndarray]]): List of tuples containing table names and embeddings.

    Returns:
        Tuple[Dict[str, Dict[str, int]], List[np.ndarray]]: Offset information and list of all embeddings.
    """
    offset = {}
    i = 0
    all_embeddings = []
    for table, embedding in embeddingfile:
        offset[table] = {'index': i, 'length': len(embedding)}
        i += 1
        all_embeddings.append(np.array(embedding, dtype=np.float32))
    return offset, all_embeddings

def get_embeddings(path: str) -> Tuple[Dict[str, Dict[str, int]], List[np.ndarray], np.ndarray]:
    """
    Load embeddings from the given path.

    Args:
        path (str): Path to the pickle file containing embeddings.

    Returns:
        Tuple[Dict[str, Dict[str, int]], List[np.ndarray], np.ndarray]: 
        Offset information, list of all embeddings, and flattened array of embeddings.
    """
    with open(path, 'rb') as f:
        embeddingfile = pickle.load(f)
    
    offset, all_embeddings = get_offset_all_embeddings(embeddingfile)
    
    flattened_array = np.concatenate(all_embeddings)
    return offset, all_embeddings, flattened_array

def get_cluster_count(linkage_matrix: np.ndarray, threshold: float) -> int:
    """
    Get the number of clusters for a given threshold.

    Args:
        linkage_matrix (np.ndarray): The linkage matrix from hierarchical clustering.
        threshold (float): The distance threshold for forming clusters.

    Returns:
        int: The number of clusters formed at the given threshold.
    """
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    num_clusters = len(np.unique(clusters))
    return num_clusters

def main():
    """
    Main function to perform hierarchical clustering on column embeddings.
    """
    logger.info("Starting column embedding clustering process")

    # Load configuration
    embedding_path = column_clustering_configs.input['embedding_source']
    datalake_path = column_clustering_configs.input['datalake_source']
    output_path = column_clustering_configs.output['result']
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.join(output_path, column_clustering_configs.output['file_name'])
    n_clusters_available = column_clustering_configs.clustering['n_clusters_available']

    # Load embeddings and create mapping
    logger.info("Loading embeddings and creating mapping")
    offset, all_embeddings, flattened_array = get_embeddings(embedding_path)
    df_map = get_mapping(datalake_path, offset)
    df_clustering_hdb = df_map.copy()

    if n_clusters_available:
        logger.info("Performing clustering with specified number of clusters")
        n_clusters = column_clustering_configs.clustering['n_clusters']
        metric = column_clustering_configs.clustering['metric']
        linkage = column_clustering_configs.clustering['linkage']
        
        start_time = time.time()
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        cluster_labels = model.fit_predict(flattened_array)
        end_time = time.time()
        
        df_clustering_hdb['cluster'] = cluster_labels
        df_clustering_hdb.to_pickle(file_name)
        
        logger.success(f"Clustering completed and saved to {file_name}")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
    else:
        logger.info("Performing full hierarchical clustering")
        start_time = time.time()
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, metric="cosine", linkage='average')
        cluster_labels = model.fit_predict(flattened_array)
        end_time = time.time()

        df_clustering_hdb['cluster'] = cluster_labels

        # Plot and save dendrogram
        plt.figure(figsize=(15, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        matrix = plot_dendrogram(model, truncate_mode='level', p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        figure_path = os.path.join(output_path, 'dendrogram.png')
        plt.savefig(figure_path)
        logger.info(f"Dendrogram saved to {figure_path}")

        # Calculate number of clusters for different thresholds
        n_cluster_array = []
        threshold_array = column_clustering_configs.clustering['experimental_thresholds']
        for threshold in threshold_array:
            count = get_cluster_count(matrix, threshold)
            logger.debug(f"Number of clusters at threshold {threshold}: {count}")
            n_cluster_array.append(count)

        # Save threshold-cluster summary
        data = {'threshold': threshold_array, 'n_cluster': n_cluster_array}
        threshold_cluster_summary = pd.DataFrame(data)
        summary_csv = os.path.join(output_path, 'threshold_n_cluster_summary.csv')
        threshold_cluster_summary.to_csv(summary_csv)
        
        logger.success(f"Clustering summary saved to {summary_csv}")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()