#!/usr/bin/env python
from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATALAKE = "ugen_v2" # "santos", "ugen_v1", "ugen_v2"
CANDIDATES = "candidates_faiss_hybrid_k_25.pkl"
METHOD = "flat-index"  # Options: "clustering", "flat-index"


COMMENT = DATALAKE + "_" + METHOD + CANDIDATES[:-4]

input = {
    "benchmark": DATALAKE,
    "query_set": "",
    "candidates": [str(BASE_DIR / "inference" / DATALAKE / "search" / CANDIDATES)],
    "table_search_task": "union", 
    "search_method": METHOD,
    "type": "search",
    "comparsion": [COMMENT],
}

benchmarks = {
    'union': ['santos', 'tus', 'tusLarge', 'pylon', 'ugen_v2'],
    'join': ['nextiajd_s', 'nextiajd_m', 'webtables_small', 'webtables_small_var2']
}

k = {
    'santos': 10,
    'tus': 60,
    'tusLarge': 60,
    'pylon': 10,
    'nextiajd_s': 10,
    'nextiajd_m': 10,
    'webtables_small': 20,
    'webtables_small_var2': 20,
    'webtables': 20,
    'ugen_v2': 10,
}

gt_paths = {
    'santos': str(BASE_DIR / "benchmarks" / "union" / "santos" / "santosUnionBenchmark.pickle"),
    'tus': str(BASE_DIR / "data" / "tus" / "small" / "tusLabeledtusUnionBenchmark"),
    'tusLarge': str(BASE_DIR / "data" / "tus" / "large" / "tusLabeledtusLargeUnionBenchmark"),
    'pylon': str(BASE_DIR / "data" / "pylon" / "all_ground_truth_general.pkl"),
    'nextiajd_s': {
        'set1': str(BASE_DIR / "data" / "nextiajd" / "testbedS" / "warpgate" / "join_dict_testbedS_warpgate.pkl"),
        'set2': str(BASE_DIR / "data" / "nextiajd" / "testbedS" / "warpgate_non-numerical" / "join_dict_testbedS_warpgate_non-numerical.pkl"),
        'set3': str(BASE_DIR / "data" / "nextiajd" / "testbedS" / "10_non-numerical" / "join_dict_testbedS_10_non-numerical.pkl")
    },
    'nextiajd_m': str(BASE_DIR / "data" / "nextiajd" / "testbedM" / "warpgate" / "join_dict_testbedM_warpgate.pkl"),
    'webtables_small': str(BASE_DIR / "data" / "lakebench" / "join" / "webtable" / "join_dict_webtable_small.pkl"),
    'webtables_small_var2': str(BASE_DIR / "data" / "lakebench" / "join" / "webtable" / "small_var2" / "join_dict_webtable_small_var2.pkl"),
    'webtables': str(BASE_DIR / "inference" / "lakebench" / "webtable" / "join_dict_final.pkl"),
    'ugen_v2': str(BASE_DIR / "benchmarks" / "union" / "ugen_v2" / "groundtruth.pickle")
}

output = {
    'path': str(BASE_DIR / "inference" / DATALAKE / "search" / "evaluation" / METHOD / COMMENT),
}