#!/usr/bin/env python
input = {
    "benchmark": "webtables",
    "query_set": "",
    "candidates": ['/home/almutawa/inference/search/lakebench/webtable/candidates_0.pkll'],
    "table_search_task": "join", 
    "search_method": 'faiss',
    "type": "search",
    "comparsion":['faiss']
}

benchmarks = {
    'union': ['santos','tus','tusLarge','pylon'],
    'join':['nextiajd_s','nextiajd_m','webtables_small','webtables_small_var2'] ##adjust 
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
    'webtables':20
}

gt_paths = {'santos':'/home/almutawa/starmie/data/santos/santosUnionBenchmark.pickle',
      'tus':'/Users/alaaalmutawa/Documents/Thesis/table-union-search-benchmark/tus/small/tusLabeledtusUnionBenchmark',
      'tusLarge':'/Users/alaaalmutawa/Documents/Thesis/table-union-search-benchmark/tus/large/tusLabeledtusLargeUnionBenchmark',
      'pylon':'/Users/alaaalmutawa/Documents/Thesis/pylon/all_ground_truth_general.pkl',
      'nextiajd_s': {'set1':'/Users/alaaalmutawa/Documents/Thesis/nextiajd/testbedS/warpgate/join_dict_testbedS_warpgate.pkl','set2':'/Users/alaaalmutawa/Documents/Thesis/nextiajd/testbedS/warpgate_non-numerical/join_dict_testbedS_warpgate_non-numerical.pkl','set3':'/Users/alaaalmutawa/Documents/Thesis/nextiajd/testbedS/10_non-numerical/join_dict_testbedS_10_non-numerical.pkl'},
      'nextiajd_m': '/Users/alaaalmutawa/Documents/Thesis/nextiajd/testbedM/warpgate/join_dict_testbedM_warpgate.pkl',
       'webtables_small':'/Users/alaaalmutawa/Documents/Thesis/lakebench/join/webtable/join_dict_webtable_small.pkl',
        'webtables_small_var2':'/Users/alaaalmutawa/Documents/Thesis/lakebench/join/webtable/small_var2/join_dict_webtable_small_var2.pkl',
        'webtables':'/home/almutawa/inference/inference/lakebench/webtable/join_dict_final.pkl'
}

output = {
    'path': '/Users/alaaalmutawa/Documents/Thesis/hytrel/hypergraph-tabular-lm/inference/inference/lakebench/webtable/evaluation',
}
