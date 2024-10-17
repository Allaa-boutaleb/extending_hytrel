import os
import shutil
from typing import Dict, List, Tuple, Any
import time
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
import hdbscan
from loguru import logger
from sklearn.impute import SimpleImputer


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

def get_next_run_id(base_output_path: str) -> int:
    existing_runs = [int(d.split('_')[1]) for d in os.listdir(base_output_path) if d.startswith('run_') and os.path.isdir(os.path.join(base_output_path, d))]
    return max(existing_runs + [0]) + 1

def save_config(config, output_path: str, run_id: int):
    config_file_name = f'config_{run_id}.py'
    config_path = os.path.join(output_path, config_file_name)
    shutil.copy2(column_clustering_configs.__file__, config_path)
    logger.info(f"Configuration saved to {config_path}")

def perform_hierarchical_clustering(flattened_array: np.ndarray, df_clustering: pd.DataFrame, output_path: str, file_name: str):
    logger.info("Performing hierarchical clustering")
    
    # Preprocess: Remove rows with NaN values
    valid_indices = ~np.isnan(flattened_array).any(axis=1)
    flattened_array_clean = flattened_array[valid_indices]
    df_clustering_clean = df_clustering[valid_indices].reset_index(drop=True)
    
    if column_clustering_configs.clustering['n_clusters_available']:
        n_clusters = column_clustering_configs.clustering['n_clusters']
        metric = column_clustering_configs.clustering['hierarchical_metric']
        linkage = column_clustering_configs.clustering['linkage']
        
        start_time = time.time()
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        cluster_labels = model.fit_predict(flattened_array_clean)
        end_time = time.time()
        
        df_clustering_clean['cluster'] = cluster_labels
        df_clustering_clean.to_pickle(file_name)
        
        logger.success(f"Clustering completed and saved to {file_name}")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
    else:
        start_time = time.time()
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric=column_clustering_configs.clustering['hierarchical_metric'], linkage=column_clustering_configs.clustering['linkage'])
        model = model.fit(flattened_array)
        end_time = time.time()

        # Plot and save dendrogram
        plt.figure(figsize=(15, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        matrix = plot_dendrogram(model, truncate_mode='level', p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        figure_path = os.path.join(output_path, 'dendrogram.png')
        plt.savefig(figure_path)
        plt.close()
        logger.info(f"Dendrogram saved to {figure_path}")

        # Calculate number of clusters for different thresholds
        n_cluster_array = []
        threshold_array = column_clustering_configs.clustering['experimental_thresholds']
        for threshold in threshold_array:
            clusters = fcluster(matrix, threshold, criterion='distance')
            df_clustering['cluster'] = clusters
            count = len(np.unique(clusters))
            logger.debug(f"Number of clusters at threshold {threshold}: {count}")
            n_cluster_array.append(count)

        # Save threshold-cluster summary
        data = {'threshold': threshold_array, 'n_cluster': n_cluster_array}
        threshold_cluster_summary = pd.DataFrame(data)
        summary_csv = os.path.join(output_path, 'threshold_n_cluster_summary.csv')
        threshold_cluster_summary.to_csv(summary_csv)
        
        # Save clustering results
        df_clustering.to_pickle(file_name)
        
        logger.success(f"Clustering summary saved to {summary_csv}")
        logger.success(f"Clustering results saved to {file_name}")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")

def perform_hdbscan_clustering(flattened_array: np.ndarray, df_clustering: pd.DataFrame, output_path: str, file_name: str):
    logger.info("Performing HDBSCAN clustering")
    
    min_cluster_size = column_clustering_configs.clustering['min_cluster_size']
    min_samples = column_clustering_configs.clustering['min_samples']
    cluster_selection_epsilon = column_clustering_configs.clustering['cluster_selection_epsilon']
    metric = column_clustering_configs.clustering['hdbscan_metric']

    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                metric=metric)
    cluster_labels = clusterer.fit_predict(flattened_array)
    end_time = time.time()

    df_clustering['cluster'] = cluster_labels
    df_clustering.to_pickle(file_name)

    # Generate and save HDBSCAN plot
    plt.figure(figsize=(10, 8))
    clusterer.condensed_tree_.plot()
    plt.title('HDBSCAN Condensed Cluster Tree')
    figure_path = os.path.join(output_path, 'hdbscan_tree.png')
    plt.savefig(figure_path)
    plt.close()
    logger.info(f"HDBSCAN tree saved to {figure_path}")

    # Save clustering summary
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    summary = {
        'n_clusters': n_clusters,
        'n_noise_points': list(cluster_labels).count(-1),
        'execution_time': end_time - start_time
    }
    summary_path = os.path.join(output_path, 'hdbscan_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.success(f"HDBSCAN clustering completed and saved to {file_name}")
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")

def main():
    logger.info("Starting column embedding clustering process")

    # Load configuration
    embedding_path = column_clustering_configs.input['embedding_source']
    datalake_path = column_clustering_configs.input['datalake_source']
    base_output_path = column_clustering_configs.output['result']
    os.makedirs(base_output_path, exist_ok=True)

    # Get the next run ID
    run_id = get_next_run_id(base_output_path)

    # Create a new directory for this run
    datalake_name = column_clustering_configs.input['datalake']
    clustering_method = column_clustering_configs.clustering['method']
    run_dir_name = f'run_{run_id}_{clustering_method}_{datalake_name}'
    output_path = os.path.join(base_output_path, run_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # Generate dynamic file name
    file_name = f'clustering_results.pkl'
    file_path = os.path.join(output_path, file_name)

    # Load embeddings and create mapping
    logger.info("Loading embeddings and creating mapping")
    offset, all_embeddings, flattened_array = get_embeddings(embedding_path)
    df_map = get_mapping(datalake_path, offset)
    df_clustering = df_map.copy()

    try:
        if clustering_method == "hierarchical":
            perform_hierarchical_clustering(flattened_array, df_clustering, output_path, file_path)
        elif clustering_method == "hdbscan":
            perform_hdbscan_clustering(flattened_array, df_clustering, output_path, file_path)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")

        # Save the configuration only if clustering completes successfully
        save_config(column_clustering_configs, output_path, run_id)
        logger.success(f"Clustering completed successfully. Results saved in {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during clustering: {str(e)}")
        # Remove the created directory if clustering fails
        shutil.rmtree(output_path)
        logger.info(f"Removed incomplete run directory: {output_path}")

if __name__ == "__main__":
    main()