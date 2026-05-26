#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/define.hpp"
#include <thread>
#include "../../hnswlib/IO.hpp"
#include <filesystem>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <sstream>

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

void generate_ground_truth(const FloatRowMat &data, const FloatRowMat &query, hnswlib::SpaceInterface<float> *space, UintRowMat &gt, int k) {
    std::cout << "--- Generating Ground Truth (k=" << k << ") ---" << std::endl;
    gt.resize(query.rows(), k);
    hnswlib::DISTFUNC<float> dist_func = space->get_dist_func();
    void* dist_func_param = space->get_dist_func_param();

    for (int i = 0; i < query.rows(); ++i) {
        std::vector<std::pair<float, unsigned int>> distances;
        for (int j = 0; j < data.rows(); ++j) {
            float dist = dist_func(query.row(i).data(), data.row(j).data(), dist_func_param);
            distances.push_back({dist, (unsigned int)j});
        }

        std::sort(distances.begin(), distances.end());

        for (int j = 0; j < k; ++j) {
            gt(i, j) = distances[j].second;
        }
    }
    std::cout << "--- Ground Truth Generation Complete ---" << std::endl << std::endl;
}

// Calculates the distance between two documents (sets of vectors) by summing the minimum distances
// between their constituent vectors.
float MinDist(const FloatRowMat &query_vectors, const FloatRowMat &data_vectors, int dim,
              size_t query_chunk_size,  size_t data_chunk_size,
              size_t query_offset,      size_t data_offset,
              hnswlib::SpaceInterface<float> *space) {
    float total_min_dist = 0.0f;
    hnswlib::DISTFUNC<float> dist_func = space->get_dist_func();
    void* dist_func_param = space->get_dist_func_param();

    // For each vector in the query document
    for (size_t q_idx = 0; q_idx < query_chunk_size; ++q_idx) {
        const float* q_vec = query_vectors.row(query_offset + q_idx).data();
        float min_dist_for_q = std::numeric_limits<float>::infinity();

        // Find the closest vector in the data document
        for (size_t d_idx = 0; d_idx < data_chunk_size; ++d_idx) {
             const float* d_vec = data_vectors.row(data_offset + d_idx).data();
             float dist = dist_func(q_vec, d_vec, dist_func_param);
             if (dist < min_dist_for_q) {
                 min_dist_for_q = dist;
             }
        }
        total_min_dist += min_dist_for_q;
    }
    return total_min_dist;
}

float MinDistSameParent(const FloatRowMat &query_vectors, const FloatRowMat &data_vectors, int dim,
    size_t query_chunk_size, size_t query_offset,      const std::vector<hnswlib::labeltype>& same_parent_ids,
    hnswlib::SpaceInterface<float> *space) {
    float total_min_dist = 0.0f;
    hnswlib::DISTFUNC<float> dist_func = space->get_dist_func();
    void* dist_func_param = space->get_dist_func_param();

    // For each vector in the query document
    for (size_t q_idx = 0; q_idx < query_chunk_size; ++q_idx) {
        const float* q_vec = query_vectors.row(query_offset + q_idx).data();
        float min_dist_for_q = std::numeric_limits<float>::infinity();

        // Find the closest vector among provided same-parent ids
        for (hnswlib::labeltype id : same_parent_ids) {
            const float* d_vec = data_vectors.row(id).data();
            float dist = dist_func(q_vec, d_vec, dist_func_param);
            if (dist < min_dist_for_q) {
                min_dist_for_q = dist;
            }
        }
        total_min_dist += min_dist_for_q;
    }
    return total_min_dist;
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <DATASET>" << std::endl;
        return 1;
    }    int dim;               // Dimension of the elements
    int M = 256;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 50;       // Number of threads for operations with index
    const char* DATASET = argv[1];

    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char result_file[500];
    char per_query_file[500];
    char per_query_file_skipping[500];
    char data_file_chunks[500];
    char query_file_chunks[500];

    sprintf(data_file, "/data1/ali/data/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(data_file_chunks, "/data1/ali/data/%s/%s_base_chunks.fvecs", DATASET, DATASET);
    sprintf(query_file, "/data1/ali/data/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(query_file_chunks, "/data1/ali/data/%s/%s_query_chunks.fvecs", DATASET, DATASET);
    sprintf(gt_file, "/data1/ali/data/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    // sprintf(gt_file, "/data1/ali/data/%s/%s_groundtruth.ivecs");
    sprintf(result_file, "/home/ali/hnswlib/results/%s_hnsw.csv", DATASET);
    sprintf(per_query_file, "/home/ali/hnswlib/results/%s_hnsw_per_query.csv", DATASET);
    sprintf(per_query_file_skipping, "/home/ali/hnswlib/results/%s_hnsw_per_query_skipping.csv", DATASET);

	FloatRowMat data_vecs;
    FloatRowMat data_chunks;
    FloatRowMat query_vecs;
    FloatRowMat query_chunks;
    UintRowMat gt_vecs;

    load_vecs<float, FloatRowMat>(data_file, data_vecs);
    load_vecs<float, FloatRowMat>(data_file_chunks, data_chunks);
    load_vecs<float, FloatRowMat>(query_file, query_vecs);
    load_vecs<float, FloatRowMat>(query_file_chunks, query_chunks);
    load_vecs<PID, UintRowMat>(gt_file, gt_vecs);

    size_t N = data_vecs.rows();
    dim = data_vecs.cols();
    size_t NQ = query_vecs.rows();
    size_t NQ_chunks = query_chunks.rows();
    std::vector<int> k_values = {1, 5, 10, 20, 50, 100};
    // std::vector<int> k_values = {1};
    int max_k = *std::max_element(k_values.begin(), k_values.end());

	std::cout << "data loaded\n";
    std::cout << "\tN: " << N <<  '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';

    // Initing index
    hnswlib::InnerProductSpace space(dim);

    // Generate random data
    // std::mt19937 rng;
    // rng.seed(47);
    // std::uniform_real_distribution<> distrib_real;
    // float* data = new float[dim * max_elements];
    // for (int i = 0; i < dim * max_elements; i++) {
    //     data[i] = distrib_real(rng);
    // }
    // generate_ground_truth(data_vecs, query_vecs, &space, gt_vecs, max_k);

    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, N, M, ef_construction);

    // Create a mapping from vector index to parent ID based on chunks
    std::vector<hnswlib::labeltype> vector_to_parent_map(N);
    size_t vector_offset = 0;
    for (int i = 0; i < data_chunks.rows(); ++i) {
        size_t chunk_size = data_chunks(i, 0);
        for (size_t j = 0; j < chunk_size; ++j) {
            size_t vector_index = vector_offset + j;
            if (vector_index >= N) {
                throw std::runtime_error("Sum of chunks exceeds the total number of vectors.");
            }
            vector_to_parent_map[vector_index] = i;
        }
        vector_offset += chunk_size;
    }
    if (vector_offset != N) {
        throw std::runtime_error("Sum of chunks does not match the total number of vectors.");
    }

    // Pre-compute parent document offsets and sizes for both data and queries
    std::vector<std::pair<size_t, size_t>> data_parent_info;
    size_t current_offset = 0;
    for (int i = 0; i < data_chunks.rows(); ++i) {
        size_t chunk_size = data_chunks(i, 0);
        data_parent_info.push_back({current_offset, chunk_size});
        current_offset += chunk_size;
    }

    std::vector<std::pair<size_t, size_t>> query_parent_info;
    current_offset = 0;
    for (int i = 0; i < query_chunks.rows(); ++i) {
        size_t chunk_size = query_chunks(i, 0);
        query_parent_info.push_back({current_offset, chunk_size});
        current_offset += chunk_size;
    }
    std::cout << "Offsets are computed" << std::endl;

    // Add data to index
    StopW stopw_indexing;
    ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {
        // std::cout << "Adding vector " << row << " to index" << std::endl;
        hnswlib::labeltype parent_id = vector_to_parent_map[row];
        alg_hnsw->addPoint((void*)(data_vecs.row(row).data()), row, parent_id);
    });
    float indexing_time_s = stopw_indexing.getElapsedTimeMicro() / 1e6;
    std::cout << "Indexing time: " << indexing_time_s << " s" << std::endl;

    // Query the elements for themselves and measure recall
    std::vector<hnswlib::labeltype> neighbors(N);
    ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_vecs.row(row).data(), 1);
        hnswlib::labeltype label = std::get<1>(result.top());
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < N; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / (float)N;
    std::cout << "Self-Recall: " << recall << "\n";

    // Open the results file for writing
    std::ofstream result_output(result_file);
    if (!result_output.is_open()) {
        std::cerr << "Failed to open result file: " << result_file << std::endl;
        return 1;
    }
    result_output << "k,avgRecall,avgHit,avgPrecision,avgQueryTimeUs,QPS" << std::endl;

    // Query the k-NN for the query set and measure recall for different K
    for (int k : k_values) {
        float recall_sum = 0.0f;    // sum of per-query Recall@k
        float hit_sum = 0.0f;       // sum of per-query Hit@k
        float precision_sum = 0.0f; // sum of per-query Precision@k
        float total_query_time_us = 0.0f; // sum of per-query timing
        
        // Loop over query DOCUMENTS (parents)
        for (int i = 0; i < NQ_chunks; i++) {
            StopW query_stopw; // Timing for individual query
            // Get query document info from pre-computed table
            size_t query_offset = query_parent_info[i].first;
            size_t query_chunk_size = query_parent_info[i].second;
            std::unordered_set<hnswlib::labeltype> candidate_parent_ids_set;
            for (size_t j = 0; j < query_chunk_size; ++j) {
                std::ostringstream vector_oss;
                std::vector<hnswlib::labeltype> candidate_parent_ids;
                auto initial_results = alg_hnsw->searchKnnSkippingDuplicates(query_vecs.row(query_offset + j).data(), k);
                while (!initial_results.empty()) {
                    // candidate_parent_ids.insert(std::get<2>(initial_results.top()));
                    auto t = initial_results.top();
                    initial_results.pop();

                    hnswlib::labeltype vec_id = std::get<1>(t);
                    hnswlib::labeltype parent_id = std::get<2>(t);
                    candidate_parent_ids.push_back(parent_id);
                    candidate_parent_ids_set.insert(parent_id);
                }
            }
            std::vector<std::pair<float, hnswlib::labeltype>> reranked_parents;
            for (const auto& parent_id : candidate_parent_ids_set) {
                size_t data_offset = data_parent_info[parent_id].first;
                size_t data_chunk_size = data_parent_info[parent_id].second;
                
                float distance = MinDist(query_vecs, data_vecs, dim,
                                         query_chunk_size, data_chunk_size,
                                         query_offset, data_offset, &space);
                reranked_parents.push_back({distance, parent_id});
            }
            // if (reranked_parents.empty()) continue;

            // Sort by distance to find the best parent document
            std::sort(reranked_parents.begin(), reranked_parents.end());

            float query_time_us = query_stopw.getElapsedTimeMicro();
            total_query_time_us += query_time_us;
    
            // Stage 3: Evaluation with multiple GTs
            std::unordered_set<hnswlib::labeltype> predicted_parent_ids;
            int num_predicted = 0;
            for (int j = 0; j < k && j < reranked_parents.size(); ++j) {
                predicted_parent_ids.insert(reranked_parents[j].second);
                num_predicted++;
            }
    
            int num_gt = gt_vecs.cols();
            int correct_matches = 0;
            int valid_gt = 0;
    
            for (int g = 0; g < num_gt; ++g) {
                hnswlib::labeltype gt_parent_id = gt_vecs(i, g);
                if (gt_parent_id == UINT_MAX){
                    // std::cout << "Invalid GT parent ID: " << gt_parent_id << std::endl;
                    continue;
                }
                valid_gt++;
                if (predicted_parent_ids.count(gt_parent_id)) {
                    correct_matches++;
                }
            }
    
            if (valid_gt > 0 && num_predicted > 0) {
                recall_sum += static_cast<float>(correct_matches) / valid_gt;            // Recall@k
                if (correct_matches > 0) hit_sum += 1.0f;                                // Hit@k
                precision_sum += static_cast<float>(correct_matches) / num_predicted;    // Precision@k
            }
            
            // Add timing for this individual query
        }
        float avg_query_time_us = total_query_time_us / static_cast<float>(NQ_chunks);
        float qps = NQ_chunks / (total_query_time_us / 1e6); // QPS based on total time for all queries
    
        float avg_recall_knn = recall_sum / static_cast<float>(NQ_chunks);
        float avg_hit_knn = hit_sum / static_cast<float>(NQ_chunks);
        float avg_precision_knn = precision_sum / static_cast<float>(NQ_chunks);
    
        std::cout << "k-NN Document Recall@" << k << ": " << avg_recall_knn
                  << ", Precision@" << k << ": " << avg_precision_knn
                  << ", Hit@" << k << ": " << avg_hit_knn
                  << ", Avg Query Time: " << avg_query_time_us << " μs"
                  << ", QPS: " << qps << "\n";
    
        result_output << k << "," << avg_recall_knn << "," << avg_hit_knn 
                      << "," << avg_precision_knn << "," << avg_query_time_us 
                      << "," << qps << std::endl;
    }

    result_output.close();

    // delete[] data;
    delete alg_hnsw;
    return 0;
}