from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

input = {
    "datalake": "lakebench",
    "datalake_source": str(BASE_DIR / "data" / "santos" / "datalake"),
    "embedding_source": str(BASE_DIR / "inference" / "santos" / "vectors"),
    'embedding_source_distributed': True, 
    "embedding_query_source": str(BASE_DIR / "inference" / "santos" / "vectors" / "hytrel_query_columns_0.pkl"),
    "downstream_task": "join",  # This dictates the format of the saved embeddings
    "method": 'hnsw'  # Options: 'faiss_hnsw' or 'faiss_flat'
}

multiple_vector_dir = {  # In case of distributed processing of the embeddings
    'index': ['001', '002', '003', '004', '005', '006'],
    'subfolder': 'vectors',
    'file_name': 'hytrel_dataset_columns_0.pkl'
}

clustering = {
    "cluster_assignment": str(BASE_DIR / "inference" / "santos" / "clustering" / 
    "run_1_hdbscan_santos" / "clustering_results.pkl"),
}

union_faiss = {
    'compress_method': 'max'
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
    'candidates': 'candidates_hdbscan_default.pkl'
}