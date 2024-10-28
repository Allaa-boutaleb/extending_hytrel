from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATALAKE = "ugen_v2" # "santos", "ugen_v1", "ugen_v2"
DOWNSTREAM_TASK = "union"
METHOD = "flat-index"  # Options: "clustering", "flat-index"

CLUSTERING_METHOD = "hdbscan" # Options: "hierarchical" or "hdbscan"
CLUSTERING_RUN = "run_"+ str(3) + f"_{CLUSTERING_METHOD}_" + DATALAKE

OUTPUT = "candidates_"+ "faiss_hybrid_k_25" + ".pkl"

input = {
    "datalake": DATALAKE,
    "datalake_source": str(BASE_DIR / "data" / DATALAKE / "datalake"),
    "embedding_source": str(BASE_DIR / "inference" / DATALAKE / "vectors" / "hytrel_datalake_columns_0.pkl"),
    'embedding_source_distributed': True, 
    "embedding_query_source": str(BASE_DIR / "inference" / DATALAKE / "vectors" / "hytrel_query_columns_0.pkl"),
    "downstream_task": DOWNSTREAM_TASK,  # This dictates the format of the saved embeddings
    "method": METHOD
}

multiple_vector_dir = {  # In case of distributed processing of the embeddings
    'index': ['001', '002', '003', '004', '005', '006'],
    'subfolder': 'vectors',
    'file_name': 'hytrel_dataset_columns_0.pkl'
}

clustering = {
    "cluster_assignment": str(BASE_DIR / "inference" / DATALAKE / "clustering" / 
    CLUSTERING_RUN / "clustering_results.pkl"),
}

union_faiss = {
    'use_two_step': True,  # Whether to use two-step efficient search
    'compress_method': 'max',  # Set to None for pure column-wise search
    'initial_filter_k': 25  # Only used if use_two_step is True
}

k = {
    'santos': 10,
    'tus': 60,
    'tusLarge': 60,
    'pylon': 10,
    'testbedS': 10,
    'testbedM': 10,
    'lakebench': 20,
    'ugen_v1': 10,
    'ugen_v2': 10,
}

output = {
    'path': str(BASE_DIR / "inference" / DATALAKE / "search"),
    'candidates': OUTPUT
}