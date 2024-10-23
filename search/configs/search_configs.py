from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

input = {
    "datalake": "santos",
    "datalake_source": str(BASE_DIR / "data" / "santos" / "datalake"),
    "embedding_source": str(BASE_DIR / "inference" / "santos" / "vectors" / "hytrel_datalake_columns_0.pkl"),
    'embedding_source_distributed': True, 
    "embedding_query_source": str(BASE_DIR / "inference" / "santos" / "vectors" / "hytrel_query_columns_0.pkl"),
    "downstream_task": "union",  # This dictates the format of the saved embeddings
    "method": 'flat-index'  # Options: 'faiss_hnsw' or 'faiss_flat'
}

multiple_vector_dir = {  # In case of distributed processing of the embeddings
    'index': ['001', '002', '003', '004', '005', '006'],
    'subfolder': 'vectors',
    'file_name': 'hytrel_dataset_columns_0.pkl'
}

clustering = {
    "cluster_assignment": str(BASE_DIR / "inference" / "santos" / "clustering" / 
    "run_7_hdbscan_santos" / "clustering_results.pkl"),
}

union_faiss = {
    'use_two_step': True,  # Whether to use two-step efficient search
    'compress_method': 'max',  # Set to None for pure column-wise search
    'initial_filter_k': 200  # Only used if use_two_step is True
}

k = {
    'santos': 10,
    'tus': 60,
    'tusLarge': 60,
    'pylon': 10,
    'testbedS': 10,
    'testbedM': 10,
    'lakebench': 20
}

output = {
    'path': str(BASE_DIR / "inference" / "santos" / "search"),
    'candidates': 'candidates_faiss_efficient_initialfilter_100_max.pkl'
}