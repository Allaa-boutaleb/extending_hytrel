import sys
from utils.utils import * ## save_metrics function
import pickle
import numpy as np
import configs.evaluation_configs as evaluation_configs

#### direct copy of the same fucntion found in checkPrecisionRecall.py in starmie's repositort ref: https://github.com/megagonlabs/starmie/tree/main
def loadDictionaryFromPickleFile(dictionaryPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictionaryPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer=open(dictionaryPath, 'rb')
    dictionary = pickle.load(filePointer)
    filePointer.close()
    return dictionary

def calc_ap(c,ap_candidate,relevant_k):
    ap_at_k = 0
    for j in range(len(ap_candidate[c])): ## sum precision at k
        ap_at_k += ap_candidate[c][j][0]*ap_candidate[c][j][1]
    if relevant_k > 0:
        ap_at_k = (1/relevant_k) * ap_at_k
    else: 
        ap_at_k = 0
    return ap_at_k

#### this function was adjusted to fit the evaluation process. It was inspired from the original function found in the file checkPrecisionRecall.py in starmie's repositort ref: https://github.com/megagonlabs/starmie/tree/main
def calc_metrics_search_results(max_k, k_range, resultFile, gtPath=None):
    ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k
    Args
        max_k: the maximum K value (e.g. for SANTOS benchmark, max_k = 10. For TUS benchmark, max_k = 60. For pylon benchmark, max_k = 10)
        k_range: step size for the K's up to max_k
        gtPath: file path to the groundtruth (dictionary in pickle format)
        resultFile: the result file to be evaluated (in pickle format)
    Return: mAP@K, P@K, R@K, fbeta, r_precision (we only use mAP@K and R@K and P@K in the analysis, the rest are for experimental purposes)
    '''
    groundtruth = loadDictionaryFromPickleFile(gtPath)
        
    # =============================================================================
    # Precision and recall
    # =============================================================================
    precision_array = []
    recall_array = []
    precision_array_mine = []
    r_precision_array = []
    fbeta_array = []
    recall_at_k = [] ## for each query
    precision_at_k = [] ## for each query 
    b = 1 ## harmonic mean of precision and recall
    map_array = []
    ap_candidate = {}
    record_at_k = {}
    for k in range(1, max_k+1):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        p_array = []
        rec = 0
        ideal_recall = []
        r_precision = []
        fbeta = [] ## fbeta score
        ap = [] ## average precision
        for table in resultFile:
            # t28 tables have less than 60 results. So, skipping them in the analysis.
            # if table.split("____",1)[0] != "t_28dc8f7610402ea7":  ### uncomment for TUS benchmark
            # print(table) ##debugging purposes 
            if table in groundtruth:
                groundtruth_set = set(groundtruth[table])
                groundtruth_set = {x for x in groundtruth_set}
                result_set = resultFile[table][:k]
                item_k = result_set[k-1:k]

                rel_k = 0
                if len(set(item_k).intersection(set(groundtruth_set))) == 1:
                    rel_k = 1
                result_set = [x for x in result_set]
                # find_intersection = true positives
                find_intersection = set(result_set).intersection(groundtruth_set) ## relevant_k
                relevant_k = len(find_intersection)
                find_diff = set(result_set).difference(groundtruth_set)
                find_diff2 = set(groundtruth_set).difference(result_set)
                tp = len(find_intersection)
                fp = len(find_diff)
                fn = len(find_diff2)
                ## for r_precision
                if len(groundtruth_set)>=k:
                    r_p = tp / k
                else:
                    r_p = tp/len(groundtruth_set)
                r_precision.append(r_p)
                r = tp/len(groundtruth_set)
                p = tp / k
                ## for ap 
                if table in ap_candidate.keys():
                    ap_candidate[table].append((p,rel_k,relevant_k))
                else: 
                    ap_candidate[table] = [(p,rel_k,relevant_k)]
                
                ap_at_k = calc_ap(table,ap_candidate,relevant_k)
                ## for fbeta
                if r + p == 0:
                    fb = 0
                else: 
                    fb = (1+np.power(b,2)) * (p * r) / ((np.power(b,2)*p )+ r)
                
                true_positive += tp
                false_positive += fp
                false_negative += fn
                rec += (tp / len(groundtruth_set))
                if tp + fp == 0:
                    p_array.append(0)
                else:
                    p_array.append(tp / (tp+fp))   
                fbeta.append(fb)
                ap.append(ap_at_k)

                ideal_recall.append(k/len(groundtruth[table]))
                if k == max_k: 
                    recall_at_k.append(r)
                    precision_at_k.append(p)
                    record_at_k[table] = {'precision':p,'recall':r}
        if (true_positive + false_positive) == 0: 
            print(f"ZERO {true_positive + false_positive}")
            precision = 0
            precision2 = 0
        else:
            precision2 = sum(p_array)/len(p_array)
            precision = true_positive / (true_positive + false_positive)
        recall = rec/len(resultFile)
        precision_array.append(precision)
        precision_array_mine.append(precision2)
        r_precision_array.append(sum(r_precision)/len(r_precision))
        recall_array.append(recall)
        fbeta_array.append(np.mean(fbeta))
        map_array.append(np.mean(ap))

        if k % 10 == 0:
            print(k, "IDEAL RECALL:", sum(ideal_recall)/len(ideal_recall))
    used_k = [k_range]
    if max_k >k_range:
        for i in range(k_range * 2, max_k+1, k_range):
            used_k.append(i)
    print("--------------------------")
    for k in used_k:
        print("Precision at k = ",k,"=", precision_array[k-1])
        # print("Precision (mine) at k = ",k,"=", precision_array_mine[k-1]) ##used for debugging
        print("Recall at k = ",k,"=", recall_array[k-1])
        print("--------------------------")
    
    map_sum = 0
    for k in range(0, max_k):
        map_sum += precision_array[k]
    mean_avg_pr = map_sum/max_k
    print("The mean average precision is:", mean_avg_pr)


    return mean_avg_pr, precision_array, recall_array,r_precision_array,fbeta_array,recall_at_k,precision_at_k,map_array,record_at_k,used_k

## for filtering only
def lshensemble_calculate_metrics(gt, res):
    with open(gt, 'rb') as f:
        gt = pickle.load(f)
    with open(res, 'rb') as f:
        res = pickle.load(f)
    precision = 0
    recall = 0
    for query_pair in res:
        gt_tables = gt[query_pair]
        res_tables = res[query_pair]
        gt_tables = set(gt_tables)
        res_tables = set(res_tables)
        common = gt_tables.intersection(res_tables)
        if len(res_tables) == 0:
            continue
        else: 
            precision = len(common)/len(res_tables) + precision
        if len(gt_tables) == 0:
            continue
        else:
            recall = len(common)/len(gt_tables) + recall

    precision = precision/len(gt)
    recall = recall/len(gt)
    return precision, recall

def evaluate_general():
    benchmark = evaluation_configs.input['benchmark']
    if benchmark == 'nextiajd_s':
        set = evaluation_configs.input['query_set']
        gt = evaluation_configs.gt_paths[benchmark][set]
    else:
        gt = evaluation_configs.gt_paths[benchmark]
    max_k = evaluation_configs.k[benchmark]
    chart_path = evaluation_configs.output['path']
    result = evaluation_configs.input['candidates']
    method = evaluation_configs.input['comparsion']
    precision_all = []
    recall_all = []
    rp_all = []
    fbeta_all = []
    map_all = []
    distribution_recall = []
    distribution_precision = []
    record_at_k_all = []
    for i in range(len(result)):
        with open(result[i],'rb') as f: 
            candidates = pickle.load(f)

        print('Method: ',method[i],'index', max_k)
        mean_avg_pr, precision_array, recall_array,r_precision_array,fbeta_array,recall_at_k,precision_at_k,map_array,record_at_k,used_k  = calc_metrics_search_results(max_k,1,candidates, gt)
        precision_all.append(precision_array)
        print(len(precision_all))
        recall_all.append(recall_array)
        rp_all.append(r_precision_array)
        fbeta_all.append(fbeta_array)
        map_all.append(map_array)
        print(len(map_all))
        record_at_k_all.append(record_at_k)
        distribution_recall.append(recall_at_k)
        distribution_precision.append(precision_at_k)
    save_metrics(used_k,precision_all,recall_all,rp_all, fbeta_all,distribution_recall,distribution_precision,map_all,record_at_k_all,method,chart_path)

def evaluate_filtering():
    benchmark = evaluation_configs.input['benchmark']
    res = evaluation_configs.input['candidates']
    if benchmark == 'nextiajd_s':
        set = evaluation_configs.input['query_set']
        gt = evaluation_configs.gt_paths[benchmark][set]
    else:
        gt = evaluation_configs.gt_paths[benchmark]
    
    for i in range(len(res)):
        precision, recall = lshensemble_calculate_metrics(gt, res[i])
        print(f"Precision: {precision}, Recall: {recall}")

def main():
    if evaluation_configs.input['type'] == 'filtering':
        evaluate_filtering()
    else: 
        evaluate_general()
    

if __name__ == '__main__':
    main()