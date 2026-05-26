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
#include <queue>

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
    }    
    int dim;               // Dimension of the elements
    int M = 256;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 50;       // Number of threads for operations with index
    const char* DATASET = argv[1];

    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char result_file[500];
    char data_file_chunks[500];
    char query_file_chunks[500];
    sprintf(data_file, "/data/ali/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(query_file, "/data/ali/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(gt_file, "/data/ali/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(result_file, "/home/ali/hnswlib/results/%s_hnsw-raw-single.csv", DATASET);

    FloatRowMat data_vecs;
    FloatRowMat query_vecs;
    UintRowMat gt_vecs;

    load_vecs<float, FloatRowMat>(data_file, data_vecs);
    load_vecs<float, FloatRowMat>(query_file, query_vecs);
    load_vecs<PID, UintRowMat>(gt_file, gt_vecs);

    size_t N = data_vecs.rows();
    dim = data_vecs.cols();
    size_t NQ = query_vecs.rows();
    std::vector<int> k_values = {1, 5, 10, 20, 50, 100};
    int max_k = *std::max_element(k_values.begin(), k_values.end());

	std::cout << "data loaded\n";
    std::cout << "\tN: " << N <<  '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';

    // Initing index
    hnswlib::InnerProductSpace space(dim);
    // Open the results file for writing
    std::ofstream result_output(result_file);
    if (!result_output.is_open()) {
        std::cerr << "Failed to open result file: " << result_file << std::endl;
        return 1;
    }
    result_output << "k,Recall,QPS" << std::endl;
    StopW stopw;
    std::vector<size_t> correct_per_k(k_values.size(), 0);
    using ResultType = std::pair<float, hnswlib::labeltype>;
    hnswlib::DISTFUNC<float> dist_func = space.get_dist_func();
    void* dist_func_param = space.get_dist_func_param();
    // Loop over query DOCUMENTS (parents)
    for (int i = 0; i < NQ; i++) {
        if (i == 1000) break;
        std::cout << "Processing query " << i << std::endl;

        std::vector<std::vector<ResultType>> thread_results(num_threads);

        ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {

            float distance = dist_func(query_vecs.row(i).data(), data_vecs.row(row).data(), dist_func_param);

            hnswlib::labeltype parent_id = static_cast<hnswlib::labeltype>(row);
            thread_results[threadId].emplace_back(distance, parent_id);
        });

        // Merge per-thread results
        std::vector<ResultType> candidates;
        for (auto &vec : thread_results) {
            candidates.insert(candidates.end(), vec.begin(), vec.end());
        }

        // Sort all by smallest distance once
        std::sort(candidates.begin(), candidates.end());

        // Compare with ground truth after processing all chunks
        hnswlib::labeltype gt_vector_label = gt_vecs(i, 0);
        if(gt_vector_label == 1881){
            continue;
        }
        for (size_t idx = 0; idx < k_values.size(); ++idx) {
            int k = k_values[idx];
            size_t upto = std::min(candidates.size(), static_cast<size_t>(k));
            bool found = false;
            for (size_t j = 0; j < upto; ++j) {
                if (candidates[j].second == gt_vector_label) {
                    found = true;
                    break;
                }
            }
            if (found) correct_per_k[idx]++;
        }
    }

    float total_time_us = stopw.getElapsedTimeMicro();
    float qps = 1000 / (total_time_us / 1e6);
    for (size_t idx = 0; idx < k_values.size(); ++idx) {
        int k = k_values[idx];
        float recall_knn = static_cast<float>(correct_per_k[idx]) / static_cast<float>(1000);
        std::cout << "k-NN Document Recall@" << k << ": " << recall_knn << ", QPS: " << qps << "\n";
        result_output << k << "," << recall_knn << "," << qps << std::endl;
    }

    result_output.close();

    return 0;
}