#!/usr/bin/env python
input = {
    "benchmark": "santos",
    "candidates": ['/home/almutawa/inference/search/santos/clustering/candidate_clusters_811_run_id_0.pkl'],
    "table_search_task": "union", 
    "search_method": 'clustering_based',
    "comparsion":['811_clusters']
}

benchmarks = {
    'union': ['santos','tus','tusLarge','pylon'],
    'join':['nextiajd_s','nextiajd_m','webtables'] ##adjust 
}
k = {
    'santos': 10,
    'tus': 60,
    'tusLarge': 60,
    'pylon': 10
}

gt_paths = {'santos':'/home/almutawa/starmie/data/santos/santosUnionBenchmark.pickle',
      'tus':'/Users/alaaalmutawa/Documents/Thesis/table-union-search-benchmark/tus/small/tusLabeledtusUnionBenchmark',
      'tusLarge':'/Users/alaaalmutawa/Documents/Thesis/table-union-search-benchmark/tus/large/tusLabeledtusLargeUnionBenchmark',
      'pylon':'/Users/alaaalmutawa/Documents/Thesis/pylon/all_ground_truth_general.pkl'
}

output = {
    'path': '/home/almutawa/inference/search/santos/clustering/evaluation/harts_excel',
}
