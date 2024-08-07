import configs.column_clustering as column_clustering_configs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from scipy.cluster.hierarchy import fcluster
import time 
import os
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

def get_mapping(path,offset):
    columns = ['index','dataset','column_name']
    df_map = pd.DataFrame(columns=columns)

    for i in offset: 
        df_temp = pd.read_csv(f'{path}/{i}')
        columns = df_temp.columns.tolist()
        for j in range(offset[i]['length']):        
            data = {'index':offset[i]['index'],'dataset':i,'column_index':int(j),'column_name':columns[j],'column_name_lower':columns[j].lower()}
            df_map = df_map.append(data, ignore_index=True)  
    return df_map

def get_offset_all_embeddings(embeddingfile):
    offset = {}
    i = 0
    all_embeddings = []
    for table, embedding in embeddingfile: 
        offset[table] = {'index':i , 'length': len(embedding)}
        i = i + 1
        all_embeddings.append(np.array(embedding, dtype=np.float32))
    return offset, all_embeddings

def get_embeddings(path):
    with open(path, 'rb') as f:
        embeddingfile = pickle.load(f)
    
    offset, all_embeddings = get_offset_all_embeddings(embeddingfile)
    
    flattened_array = np.concatenate(all_embeddings)
    return offset, all_embeddings, flattened_array


def get_cluster_count(linkage_matrix, threshold):
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    num_clusters = len(np.unique(clusters))
    return num_clusters

def main():
    embedding_path = column_clustering_configs.input['embedding_source']
    datalake_path = column_clustering_configs.input['datalake_source']
    output_path = column_clustering_configs.output['result']
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.join(output_path,column_clustering_configs.output['file_name'])
    n_clusters_available = column_clustering_configs.clustering['n_clusters_available']
    offset, all_embeddings, flattened_array = get_embeddings(embedding_path)
    df_map = get_mapping(datalake_path,offset)
    df_clustering_hdb = df_map.copy()
    print('=============== clustering module started ===============')
    if n_clusters_available: 
        print('=============== n_cluster specified ===============')
        n_clusters = column_clustering_configs.clustering['n_clusters']
        affinity = column_clustering_configs.clustering['affinity']
        linkage = column_clustering_configs.clustering['linkage']
        start_time = time.time()
        model = AgglomerativeClustering(n_clusters=n_clusters,affinity=affinity,linkage=linkage)
        cluster_labels = model.fit_predict(flattened_array) 
        end_time = time.time()
        df_clustering_hdb['cluster'] = cluster_labels
        df_clustering_hdb.to_pickle(file_name)
        print('=============== saved ===============')
        print(f'path: {file_name}')
        print(f'execution time: {end_time - start_time} seconds')
    else:
        print('=============== n_cluster not specified ===============')
        start_time = time.time()
        model = AgglomerativeClustering(n_clusters=None,distance_threshold=0.1,affinity="cosine",linkage='average')
        end_time = time.time()
        cluster_labels = model.fit_predict(flattened_array)
        df_clustering_hdb['cluster'] = cluster_labels
        plt.figure(figsize=(15, 10))  
        plt.title('Hierarchical Clustering Dendrogram')
        matrix = plot_dendrogram(model, truncate_mode='level', p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        figure_path = os.path.join(output_path,'dendrogram.png')
        plt.savefig(figure_path)
        n_cluster_array = []
        threshold_array = column_clustering_configs.clustering['experimental_thresholds']
        for t in threshold_array:
            threshold = t
            count = get_cluster_count(matrix, threshold)
            print(f"Number of clusters at threshold {threshold}: {count}")
            n_cluster_array.append(count)
        data = {'threshold': threshold_array, 
                'n_cluster': n_cluster_array}
        threshold_cluster_summary = pd.DataFrame(data)
        summary_csv = os.path.join(output_path,'threshold_n_cluster_summary.csv')
        threshold_cluster_summary.to_csv(summary_csv)
        print('=============== saved ===============')
        print(f'path: {summary_csv}')
        print(f'execution time: {end_time - start_time} seconds')
    
if __name__ == "__main__":
    main()

