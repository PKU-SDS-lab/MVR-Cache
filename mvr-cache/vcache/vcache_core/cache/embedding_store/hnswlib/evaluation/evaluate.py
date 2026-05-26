import csv
import logging
from typing import Dict, List, Tuple

import numpy as np
import pytrec_eval

def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    仿照 beir.retrieval.evaluation.EvaluateRetrieval.evaluate 编写的评估函数。
    """
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval_metric in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k, v in eval_metric.items():
            logging.info(f"{k}: {v:.4f}")

    return ndcg, _map, recall, precision


def load_gt(gt_path: str) -> Dict[str, Dict[str, int]]:
    """
    加载 ground-truth npy 文件并转换为 pytrec_eval 所需的 qrels 格式。
    查询ID将使用其在npy文件中的索引（0, 1, 2, ...）。
    """
    gt_data = np.load(gt_path, allow_pickle=True)
    qrels = {}
    for i, gt_list in enumerate(gt_data):
        query_id = str(i)
        qrels[query_id] = {}
        for passage_id in gt_list:
            qrels[query_id][str(passage_id)] = 1  # 假设相关性得分为1
    return qrels


def load_results(results_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载检索结果的tsv文件并转换为 pytrec_eval 所需的 results 格式。
    - 如果文件有4列 (query_id, passage_id, rank, score)，则使用第四列的分数。
    - 如果文件只有3列 (query_id, passage_id, rank)，则使用 1/rank 作为分数。
    """
    results = {}
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row: continue # Skip empty lines

            query_id, passage_id = row[0], row[1]
            
            if query_id not in results:
                results[query_id] = {}
            
            # 判断使用真实分数还是生成代理分数
            if len(row) == 4:
                score = float(row[3])
            elif len(row) == 3:
                rank = int(row[2])
                score = 1.0 / rank
            else:
                logging.warning(f"Skipping malformed line with {len(row)} columns: {row}")
                continue
            
            results[query_id][passage_id] = score
            
    return results


if __name__ == "__main__":
    # 配置日志记录
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # --- 文件路径配置 ---
    GT_FILE_PATH = "/home/ali/Extended-RaBitQ-main/evaluation/ground-truth/openai1M/gt.npy"
    RESULTS_FILE_PATH = "/home/ali/Extended-RaBitQ-main/evaluation/result/multi-hnsw.tsv" # plaid multi-hnsw
    # RESULTS_FILE_PATH = "./result/plaid.tsv"
    
    # --- 评估参数 ---
    K_VALUES = [1, 3, 5, 10, 20, 50, 100]

    logging.info(f"Loading ground-truth file {GT_FILE_PATH}...")
    qrels_data = load_gt(GT_FILE_PATH)
    
    logging.info(f"Loading retrieval results file{RESULTS_FILE_PATH}...")
    results_data = load_results(RESULTS_FILE_PATH)

    logging.info("Starting evaluation...")
    evaluate(qrels=qrels_data, results=results_data, k_values=K_VALUES)
    logging.info("Evaluation finished.")