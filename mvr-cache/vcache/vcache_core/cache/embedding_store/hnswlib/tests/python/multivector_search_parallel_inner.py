import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import hnswlib

PAD_ID = np.iinfo(np.uint64).max


def load_fvecs(path: str) -> np.ndarray:
    """FAISS .fvecs -> float32 matrix (n, d)."""
    x = np.fromfile(path, dtype=np.int32)
    if x.size == 0:
        raise RuntimeError(f"Empty fvecs file: {path}")
    d = int(x[0])
    x = x.reshape(-1, d + 1)
    return x[:, 1:].view(np.float32)


def load_ivecs(path: str) -> np.ndarray:
    """FAISS .ivecs -> int32 matrix (n, d)."""
    x = np.fromfile(path, dtype=np.int32)
    if x.size == 0:
        raise RuntimeError(f"Empty ivecs file: {path}")
    d = int(x[0])
    x = x.reshape(-1, d + 1)
    return x[:, 1:]


def min_dist_ip_sum(query_doc: np.ndarray, data_doc: np.ndarray) -> float:
    """
    Python equivalent of C++ MinDist() for InnerProductSpace when dist = -dot.
    total = sum_q min_d (-q·d) = -sum_q max_d (q·d)
    """
    sims = query_doc @ data_doc.T
    return float((-sims.max(axis=1)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Python port of tests/cpp/multivector_search_parallel_inner.cpp"
    )
    ap.add_argument("DATASET", help="Dataset name (same positional arg as the C++ binary)")
    ap.add_argument("--fix_gt_parent_check", action="store_true",
                    help="C++ compares predicted parent IDs against gt vector ID (likely a bug). "
                         "Enable this to map gt vector -> parent before comparing.")
    args = ap.parse_args()

    DATASET = args.DATASET

    dim = None
    M = 256
    ef_construction = 256
    num_threads = 50
    k_values = [1, 5, 10]

    data_file = f"/data/ali/{DATASET}/{DATASET}_base.fvecs"
    data_file_chunks = f"/data/ali/{DATASET}/{DATASET}_base_chunks.fvecs"
    query_file = f"/data/ali/{DATASET}/{DATASET}_query.fvecs"
    query_file_chunks = f"/data/ali/{DATASET}/{DATASET}_query_chunks.fvecs"
    gt_file = f"/data/ali/{DATASET}/{DATASET}_groundtruth.ivecs"
    result_file = f"/home/ali/hnswlib/results/{DATASET}_hnsw-half-qps.csv"

    data_vecs = load_fvecs(data_file)
    data_chunks = load_fvecs(data_file_chunks)
    query_vecs = load_fvecs(query_file)
    query_chunks = load_fvecs(query_file_chunks)
    gt_vecs = load_ivecs(gt_file)

    N = int(data_vecs.shape[0])
    dim = int(data_vecs.shape[1])
    NQ = int(query_vecs.shape[0])
    NQ_chunks = int(query_chunks.shape[0])

    print("data loaded")
    print(f"\tN: {N}")
    print("query loaded")
    print(f"\tNQ: {NQ}")

    # Initing index (Inner Product)
    space = "ip"
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=N, M=M, ef_construction=ef_construction)
    index.set_num_threads(num_threads)

    # Create a mapping from vector index to parent ID based on chunks
    vector_to_parent_map = np.empty((N,), dtype=np.uint64)
    vector_offset = 0
    for i in range(int(data_chunks.shape[0])):
        chunk_size = int(data_chunks[i, 0])
        for j in range(chunk_size):
            vector_index = vector_offset + j
            if vector_index >= N:
                raise RuntimeError("Sum of chunks exceeds the total number of vectors.")
            vector_to_parent_map[vector_index] = np.uint64(i)
        vector_offset += chunk_size
    if vector_offset != N:
        raise RuntimeError("Sum of chunks does not match the total number of vectors.")

    # Pre-compute parent document offsets and sizes for both data and queries
    data_parent_info = []
    current_offset = 0
    for i in range(int(data_chunks.shape[0])):
        chunk_size = int(data_chunks[i, 0])
        data_parent_info.append((current_offset, chunk_size))
        current_offset += chunk_size

    query_parent_info = []
    current_offset = 0
    for i in range(int(query_chunks.shape[0])):
        chunk_size = int(query_chunks[i, 0])
        query_parent_info.append((current_offset, chunk_size))
        current_offset += chunk_size
    print("Offsets are computed")

    # Add data to index
    t_index0 = time.perf_counter()
    ids = np.arange(N, dtype=np.uint64)
    index.add_items(data_vecs, ids=ids, parent_ids=vector_to_parent_map, num_threads=num_threads)
    indexing_time_s = time.perf_counter() - t_index0
    print(f"Indexing time: {indexing_time_s} s")

    index_path = f"/data/ali/{DATASET}_index_raw.bin"
    print(f"Saving index to: {index_path}")
    index.save_index(index_path)
    print("Index saved successfully")

    # Open the results file for writing
    Path("/home/ali/hnswlib/results").mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as result_output:
        result_output.write("k,Recall,QPS\n")

        # Thread pool used similarly to the C++ ThreadPool
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for k in k_values:
                correct_knn = 0.0
                t0 = time.perf_counter()

                # Loop over query DOCUMENTS (parents)
                for i in range(NQ_chunks):
                    query_offset, query_chunk_size = query_parent_info[i]

                    all_search_results = [None] * int(query_chunk_size)

                    def do_one_vector(j: int):
                        q = query_vecs[query_offset + j]
                        # Match C++ behavior: each task runs one query; keep internal threads at 1.
                        labels, dists, parents = index.knn_query_skipping_duplicates_with_parent(
                            q.reshape(1, -1), k=k, num_threads=1
                        )
                        # Convert to list[tuple(dist,label,parent)] in ascending distance order.
                        res = []
                        for t in range(k):
                            res.append((float(dists[0, t]), int(labels[0, t]), int(parents[0, t])))
                        return j, res

                    futures = [pool.submit(do_one_vector, j) for j in range(int(query_chunk_size))]
                    for fut in futures:
                        j, res = fut.result()
                        all_search_results[j] = res

                    candidate_parent_ids_set = set()
                    for j in range(int(query_chunk_size)):
                        search_results = all_search_results[j]
                        if search_results:
                            for (approx_dist, label, parent_id) in search_results:
                                if parent_id == PAD_ID:
                                    continue
                                candidate_parent_ids_set.add(parent_id)

                    candidate_parent_ids = list(candidate_parent_ids_set)

                    # Parallel compute MinDist for each candidate parent
                    reranked_parents = [None] * len(candidate_parent_ids)

                    def do_one_parent(ci: int):
                        parent_id = candidate_parent_ids[ci]
                        data_offset, data_chunk_size = data_parent_info[parent_id]
                        q_doc = query_vecs[query_offset:query_offset + query_chunk_size]
                        d_doc = data_vecs[data_offset:data_offset + data_chunk_size]
                        distance = min_dist_ip_sum(q_doc, d_doc)
                        return ci, (distance, parent_id)

                    futures2 = [pool.submit(do_one_parent, ci) for ci in range(len(candidate_parent_ids))]
                    for fut in futures2:
                        ci, val = fut.result()
                        reranked_parents[ci] = val

                    reranked_parents.sort(key=lambda x: x[0])

                    predicted_parent_ids = set()
                    for j in range(min(k, len(reranked_parents))):
                        predicted_parent_ids.add(int(reranked_parents[j][1]))

                    # Stage 3: Check against ground truth (match C++ default behavior)
                    gt_vector_label = int(gt_vecs[i, 0])
                    if args.fix_gt_parent_check:
                        gt_check_value = int(vector_to_parent_map[gt_vector_label])
                    else:
                        gt_check_value = gt_vector_label

                    if gt_check_value in predicted_parent_ids:
                        correct_knn += 1.0

                total_time_us = (time.perf_counter() - t0) * 1e6
                qps = NQ_chunks / (total_time_us / 1e6)
                recall_knn = correct_knn / float(NQ_chunks)
                print(f"k-NN Document Recall@{k}: {recall_knn}, QPS: {qps}")
                result_output.write(f"{k},{recall_knn},{qps}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

