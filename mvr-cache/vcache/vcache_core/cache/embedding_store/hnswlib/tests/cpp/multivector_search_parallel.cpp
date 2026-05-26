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
    return total_min_dist / query_chunk_size;
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
    char data_file_chunks[500];
    char query_file_chunks[500];

    sprintf(data_file, "/data/ali/clerc-large-multi/%s_base.fvecs", DATASET);
    sprintf(data_file_chunks, "/data/ali/clerc-large-multi/%s_base_chunks.fvecs", DATASET);
    sprintf(query_file, "/data/ali/clerc-large-multi/%s_query.fvecs", DATASET);
    sprintf(query_file_chunks, "/data/ali/clerc-large-multi/%s_query_chunks.fvecs", DATASET);
    sprintf(gt_file, "/data/ali/clerc-large-multi/%s_groundtruth.ivecs", DATASET);
    sprintf(result_file, "/home/ali/hnswlib/results/%s_hnsw-half-qps.csv", DATASET);
    sprintf(per_query_file, "/home/ali/hnswlib/results/%s_hnsw_per_query-half.csv", DATASET);


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

    // std::string index_path = "/data/ali/indexing/" + std::string(DATASET) + "_index_raw.bin";
    // std::cout << "Loading index from: " << index_path << std::endl;
    
    // StopW stopw_loading;
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    // float loading_time_s = stopw_loading.getElapsedTimeMicro() / 1e6;
    // std::cout << "Index loading time: " << loading_time_s << " s" << std::endl;
    // std::cout << "Index loaded successfully" << std::endl;

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
    std::string index_path = "/data/ali/indexing/" + std::string(DATASET) + "_index_raw.bin";
    std::cout << "Saving index to: " << index_path << std::endl;
    alg_hnsw->saveIndex(index_path);
    std::cout << "Index saved successfully" << std::endl;

    // Query the elements for themselves and measure recall
    // std::vector<hnswlib::labeltype> neighbors(N);
    // ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {
    //     std::priority_queue<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_vecs.row(row).data(), 1);
    //     hnswlib::labeltype label = std::get<1>(result.top());
    //     neighbors[row] = label;
    // });
    // float correct = 0;
    // for (int i = 0; i < N; i++) {
    //     hnswlib::labeltype label = neighbors[i];
    //     if (label == i) correct++;
    // }
    // float recall = correct / (float)N;
    // std::cout << "Self-Recall: " << recall << "\n";

    // Open the results file for writing
    std::ofstream result_output(result_file);
    if (!result_output.is_open()) {
        std::cerr << "Failed to open result file: " << result_file << std::endl;
        return 1;
    }
    result_output << "k,Recall,QPS" << std::endl;

    // Open the per-query results file for writing
    // std::ofstream per_query_output(per_query_file);
    // if (!per_query_output.is_open()) {
    //     std::cerr << "Failed to open per-query result file: " << per_query_file << std::endl;
    //     return 1;
    // }
    // per_query_output << "query_id,k,predicted_ids" << std::endl;

    // Pre-compute search results for all query vectors using maximum k
    // StopW stopw;
    // std::cout << "--- Pre-computing search results for all query vectors ---" << std::endl;
    StopW search_time;
    
    // Data structure to store search results: [query_vector_index] -> vector of (distance, vec_id, parent_id)
    std::vector<std::vector<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>>> all_search_results(NQ);

    for (int k : k_values) {
        float correct_knn = 0;   
        StopW stopw;
        // Parallel search for all query vectors
        std::vector<std::unordered_set<hnswlib::labeltype>> candidate_parent_ids_per_query(NQ_chunks);
        ParallelFor(0, NQ, 10, [&](size_t query_idx, size_t threadId) {
            auto results = alg_hnsw->searchKnn(query_vecs.row(query_idx).data(), k);
            
            // Store results in reverse order (since priority_queue gives largest first)
            std::vector<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>> result_vec;
            while (!results.empty()) {
                result_vec.push_back(results.top());
                results.pop();
            }
            // Reverse to get smallest distances first
            std::reverse(result_vec.begin(), result_vec.end());
            all_search_results[query_idx] = std::move(result_vec);
        });
        // Pre-compute candidate parent IDs for all query documents using parallel processing
        ParallelFor(0, NQ_chunks, num_threads, [&](size_t i, size_t threadId) {
            size_t query_offset = query_parent_info[i].first;
            size_t query_chunk_size = query_parent_info[i].second;
            
            std::unordered_set<hnswlib::labeltype>& candidate_parent_ids_set = candidate_parent_ids_per_query[i];
            
            // Collect candidate parent IDs from all vectors in this query document
            for (size_t j = 0; j < query_chunk_size; ++j) {
                size_t query_vec_idx = query_offset + j;
                const auto& search_results = all_search_results[query_vec_idx];
                
                for (const auto& result : search_results) {
                    hnswlib::labeltype parent_id = std::get<2>(result);
                    candidate_parent_ids_set.insert(parent_id);
                }
            }
        });
        std::vector<std::vector<std::pair<float, hnswlib::labeltype>>> precomputed_distances(NQ_chunks);
        // Pre-compute distances for all query documents using parallel processing
        ParallelFor(0, NQ_chunks, num_threads, [&](size_t i, size_t threadId) {
            size_t query_offset = query_parent_info[i].first;
            size_t query_chunk_size = query_parent_info[i].second;
            const std::unordered_set<hnswlib::labeltype>& candidate_parent_ids_set = candidate_parent_ids_per_query[i];
            std::vector<std::pair<float, hnswlib::labeltype>>& reranked_parents = precomputed_distances[i];
            
            // Convert set to vector for processing
            std::vector<hnswlib::labeltype> parent_ids_vec(candidate_parent_ids_set.begin(), candidate_parent_ids_set.end());
            reranked_parents.resize(parent_ids_vec.size());
            
            // Compute distances for all candidate parents
            for (size_t idx = 0; idx < parent_ids_vec.size(); idx++) {
                hnswlib::labeltype parent_id = parent_ids_vec[idx];
                size_t data_offset = data_parent_info[parent_id].first;
                size_t data_chunk_size = data_parent_info[parent_id].second;
                
                float distance = MinDist(query_vecs, data_vecs, dim,
                                        query_chunk_size, data_chunk_size,
                                        query_offset, data_offset, &space);
                reranked_parents[idx] = {distance, parent_id};
            }
            
            // Sort by distance to find the best parent document
            std::sort(reranked_parents.begin(), reranked_parents.end());
        });

        // Calculate recall for this k value
        for (size_t i = 0; i < NQ_chunks; ++i) {
            const std::vector<std::pair<float, hnswlib::labeltype>>& reranked_parents = precomputed_distances[i];
            std::unordered_set<hnswlib::labeltype> predicted_parent_ids;
            for (int j = 0; j < k && j < reranked_parents.size(); ++j) {
                predicted_parent_ids.insert(reranked_parents[j].second);
            }
            hnswlib::labeltype gt_vector_label = gt_vecs(i, 0);
            if (predicted_parent_ids.count(gt_vector_label)) {
                correct_knn++;
            }
        }
        float total_time_us = stopw.getElapsedTimeMicro();
        float qps = NQ_chunks / (total_time_us / 1e6);
        float recall_knn = correct_knn / (float)NQ_chunks;
        std::cout << "k-NN Document Recall@" << k << ": " << recall_knn << ", QPS: " << qps << "\n";
        result_output << k << "," << recall_knn << "," << qps << std::endl;
    }

    result_output.close();

    delete alg_hnsw;
    return 0;
}