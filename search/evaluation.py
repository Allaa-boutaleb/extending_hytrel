import sys
from utils.utils import save_metrics
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from loguru import logger
import configs.evaluation_configs as evaluation_configs

import sys
from utils.utils import save_metrics
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from loguru import logger
import configs.evaluation_configs as evaluation_configs

def loadDictionaryFromPickleFile(dictionaryPath: str) -> Dict:
    """Load the pickle file as a dictionary."""
    with open(dictionaryPath, 'rb') as filePointer:
        dictionary = pickle.load(filePointer)
    return dictionary

def calc_ap(c: str, ap_candidate: Dict[str, List[Tuple[float, int, int]]], relevant_k: int) -> float:
    """Calculate Average Precision for a single query."""
    ap_at_k = sum(p * rel for p, rel, _ in ap_candidate[c])
    return ap_at_k / relevant_k if relevant_k > 0 else 0

def calc_metrics_search_results(max_k: int, k_range: int, resultFile: Dict[str, List[str]], gtPath: str) -> Tuple[float, List[float], List[float], List[float], List[float], List[float], List[float], List[float], Dict[str, Dict[int, Dict[str, float]]], List[int], Dict[str, Dict[int, float]]]:
    """Calculate and log the performance metrics: MAP, Precision@k, Recall@k, etc."""
    groundtruth = loadDictionaryFromPickleFile(gtPath)
    
    precision_array, recall_array, r_precision_array, fbeta_array, map_array = [], [], [], [], []
    recall_at_k, precision_at_k = [], []
    ap_candidate: Dict[str, List[Tuple[float, int, int]]] = {}
    record_at_k: Dict[str, Dict[int, Dict[str, float]]] = {}
    ap_at_k: Dict[str, Dict[int, float]] = {}

    for k in range(1, max_k+1):
        true_positive = false_positive = false_negative = 0
        p_array, r_precision, fbeta, ap = [], [], [], []

        for table in resultFile:
            if table in groundtruth:
                groundtruth_set = set(groundtruth[table])
                result_set = set(resultFile[table][:k])
                
                tp = len(result_set.intersection(groundtruth_set))
                fp = len(result_set.difference(groundtruth_set))
                fn = len(groundtruth_set.difference(result_set))
                
                p = tp / k if k > 0 else 0
                r = tp / len(groundtruth_set) if groundtruth_set else 0

                if table not in ap_candidate:
                    ap_candidate[table] = []
                ap_candidate[table].append((p, 1 if tp > len(ap_candidate[table]) else 0, tp))
                
                ap_value = calc_ap(table, ap_candidate, len(groundtruth_set))
                
                r_p = tp / min(k, len(groundtruth_set)) if min(k, len(groundtruth_set)) > 0 else 0
                fb = (2 * p * r) / (p + r) if (p + r) > 0 else 0
                
                true_positive += tp
                false_positive += fp
                false_negative += fn
                p_array.append(p)
                r_precision.append(r_p)
                fbeta.append(fb)
                ap.append(ap_value)

                logger.info(f"Table: {table}, k: {k}, Precision@k: {p:.4f}, Recall@k: {r:.4f}, AP@k: {ap_value:.4f}")

                if table not in record_at_k:
                    record_at_k[table] = {}
                if table not in ap_at_k:
                    ap_at_k[table] = {}

                record_at_k[table][k] = {'precision': p, 'recall': r}
                ap_at_k[table][k] = ap_value

        precision = sum(p_array) / len(p_array) if p_array else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        if k in range(k_range, max_k + 1, k_range):
            recall_at_k.extend(p for table, values in record_at_k.items() if k in values for p in [values[k]['recall']])
            precision_at_k.extend(p for table, values in record_at_k.items() if k in values for p in [values[k]['precision']])

        precision_array.append(precision)
        recall_array.append(recall)
        r_precision_array.append(sum(r_precision) / len(r_precision) if r_precision else 0)  
        fbeta_array.append(sum(fbeta) / len(fbeta) if fbeta else 0)
        map_array.append(sum(ap) / len(ap) if ap else 0)

    used_k = list(range(k_range, max_k + 1, k_range))
    mean_avg_pr = sum(precision_array) / len(precision_array) if precision_array else 0

    logger.info(f"Mean Average Precision: {mean_avg_pr}")
    for k in used_k:
        logger.info(f"Precision at k = {k}: {precision_array[k-1]}")
        logger.info(f"Recall at k = {k}: {recall_array[k-1]}")

    return mean_avg_pr, precision_array, recall_array, r_precision_array, fbeta_array, recall_at_k, precision_at_k, map_array, record_at_k, used_k, ap_at_k


def lshensemble_calculate_metrics(gt: str, res: str) -> Tuple[float, float]:
    """Calculate precision and recall for LSH Ensemble results."""
    with open(gt, 'rb') as f:
        gt = pickle.load(f)
    with open(res, 'rb') as f:
        res = pickle.load(f)
    
    precision = recall = 0
    for query_pair in res:
        gt_tables = set(gt[query_pair])
        res_tables = set(res[query_pair])
        common = gt_tables.intersection(res_tables)
        
        if res_tables:
            precision += len(common) / len(res_tables)
        if gt_tables:
            recall += len(common) / len(gt_tables)

    precision /= len(gt)
    recall /= len(gt)
    return precision, recall

def evaluate_general():
    """Evaluate general search results."""
    benchmark = evaluation_configs.input['benchmark']
    gt = evaluation_configs.gt_paths[benchmark]
    if benchmark == 'nextiajd_s':
        set = evaluation_configs.input['query_set']
        gt = gt[set]
    
    max_k = evaluation_configs.k[benchmark]
    chart_path = evaluation_configs.output['path']
    result = evaluation_configs.input['candidates']
    method = evaluation_configs.input['comparsion']
    
    precision_all, recall_all, rp_all, fbeta_all, map_all = [], [], [], [], []
    distribution_recall, distribution_precision = [], []
    record_at_k_all, ap_at_k_all = [], []

    for i, res in enumerate(result):
        with open(res, 'rb') as f:
            candidates = pickle.load(f)

        logger.info(f'Method: {method[i]}, index {max_k}') 
        metrics = calc_metrics_search_results(max_k, 1, candidates, gt)
        mean_avg_pr, precision_array, recall_array, r_precision_array, fbeta_array, recall_at_k, precision_at_k, map_array, record_at_k, used_k, ap_at_k = metrics

        precision_all.append(precision_array)
        recall_all.append(recall_array)
        rp_all.append(r_precision_array)
        fbeta_all.append(fbeta_array)
        map_all.append(map_array)
        record_at_k_all.append(record_at_k)
        distribution_recall.append(recall_at_k)
        distribution_precision.append(precision_at_k)
        ap_at_k_all.append(ap_at_k)

    save_metrics(used_k, precision_all, recall_all, rp_all, fbeta_all, distribution_recall, distribution_precision, map_all, record_at_k_all, ap_at_k_all, method, chart_path)

def evaluate_filtering():
    """Evaluate filtering results."""
    benchmark = evaluation_configs.input['benchmark']
    res = evaluation_configs.input['candidates']
    gt = evaluation_configs.gt_paths[benchmark]
    if benchmark == 'nextiajd_s':
        set = evaluation_configs.input['query_set']
        gt = gt[set]
    
    for i, r in enumerate(res):
        precision, recall = lshensemble_calculate_metrics(gt, r)
        logger.info(f"Method {i+1} - Precision: {precision:.4f}, Recall: {recall:.4f}")

def main():
    """Main function to run evaluation."""
    if evaluation_configs.input['type'] == 'filtering':
        evaluate_filtering()
    else:
        evaluate_general()

if __name__ == '__main__':
    logger.add("evaluation.log", rotation="10 MB")
    main()