#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/define.hpp"
#include <thread>
#include "../../hnswlib/IO.hpp"
#include <filesystem>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <functional>
#include <atomic>
#include <exception>
#include <tuple>
#include <unordered_map>
#include <numeric>
#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>

class FunctionPool {
public:
    FunctionPool(size_t pool_size) {
        pool.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            pool.push_back(nullptr); // preallocate empty slots
        }
    }

    std::function<void()>& get() {
        std::unique_lock<std::mutex> lock(mtx);
        if (pool.empty()) {
            pool.push_back(nullptr); // fallback
        }
        auto& func = pool.back();
        pool.pop_back();
        return func;
    }

    void release(std::function<void()>& func) {
        func = nullptr; // reset lambda capture
        std::unique_lock<std::mutex> lock(mtx);
        pool.push_back(func);
    }

private:
    std::vector<std::function<void()>> pool;
    std::mutex mtx;
};


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

class ThreadPool {
    public:
        ThreadPool(size_t numThreads) : stop(false), tasks_in_flight(0) {
            workers.reserve(numThreads);
            for (size_t i = 0; i < numThreads; ++i) {
                workers.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
    
                        // Fetch task
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                            if (stop && tasks.empty()) return;
                            task = std::move(tasks.front());
                            tasks.pop();
                        }
    
                        // Execute task
                        task();
    
                        // Decrement counter and notify if last task
                        if (--tasks_in_flight == 0) {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            cv_done.notify_one();
                        }
                    }
                });
            }
        }
    
        // Submit a task (lambda or std::function)
        void enqueue(std::function<void()> func) {
            tasks_in_flight++;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                tasks.push(std::move(func));
            }
            cv.notify_one();
        }
    
        // Wait for all tasks to complete
        void wait() {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv_done.wait(lock, [this]{ return tasks_in_flight.load() == 0; });
        }
    
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            cv.notify_all();
            for (auto &thread : workers) thread.join();
        }
    
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
    
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::condition_variable cv_done;
        std::atomic<size_t> tasks_in_flight;
        bool stop;
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

#ifdef ENABLE_AVX2_IP
    auto dot4_avx2 = [&](const float* q_ptr,
                         const float* d0,
                         const float* d1,
                         const float* d2,
                         const float* d3,
                         int d) -> std::array<float,4> {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            __m256 qv = _mm256_loadu_ps(q_ptr + i);
            __m256 v0 = _mm256_loadu_ps(d0 + i);
            __m256 v1 = _mm256_loadu_ps(d1 + i);
            __m256 v2 = _mm256_loadu_ps(d2 + i);
            __m256 v3 = _mm256_loadu_ps(d3 + i);
#if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(qv, v0, acc0);
            acc1 = _mm256_fmadd_ps(qv, v1, acc1);
            acc2 = _mm256_fmadd_ps(qv, v2, acc2);
            acc3 = _mm256_fmadd_ps(qv, v3, acc3);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(qv, v0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(qv, v1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(qv, v2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(qv, v3));
#endif
        }
        // Horizontal sum of 256-bit vectors
        auto hsum256 = [](__m256 v) -> float {
            __m128 vlow  = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            vlow = _mm_add_ps(vlow, vhigh);
            __m128 shuf = _mm_movehdup_ps(vlow);
            __m128 sums = _mm_add_ps(vlow, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        };
        float s0 = hsum256(acc0);
        float s1 = hsum256(acc1);
        float s2 = hsum256(acc2);
        float s3 = hsum256(acc3);
        // Tail
        for (; i < d; ++i) {
            float qv = q_ptr[i];
            s0 += qv * d0[i];
            s1 += qv * d1[i];
            s2 += qv * d2[i];
            s3 += qv * d3[i];
        }
        return {s0, s1, s2, s3};
    };
#endif

    // For each vector in the query document
    for (size_t q_idx = 0; q_idx < query_chunk_size; ++q_idx) {
        const float* q_vec = query_vectors.row(query_offset + q_idx).data();
        float min_dist_for_q = std::numeric_limits<float>::infinity();

#ifndef ENABLE_AVX2_IP
        // Generic path: use provided distance function; unroll by 2 and prefetch
        size_t d_idx = 0;
        for (; d_idx + 2 < data_chunk_size; d_idx += 2) {
            const float* d_vec0 = data_vectors.row(data_offset + d_idx).data();
            const float* d_vec1 = data_vectors.row(data_offset + d_idx + 1).data();
            if (d_idx + 8 < data_chunk_size) {
                __builtin_prefetch(data_vectors.row(data_offset + d_idx + 8).data(), 0, 1);
            }
            float dist0 = dist_func(q_vec, d_vec0, dist_func_param);
            float dist1 = dist_func(q_vec, d_vec1, dist_func_param);
            min_dist_for_q = std::min(min_dist_for_q, dist0);
            min_dist_for_q = std::min(min_dist_for_q, dist1);
        }
        for (; d_idx < data_chunk_size; ++d_idx) {
            const float* d_vec = data_vectors.row(data_offset + d_idx).data();
            float dist = dist_func(q_vec, d_vec, dist_func_param);
            if (dist < min_dist_for_q) min_dist_for_q = dist;
        }
#else
        // Fast path for inner-product space: batch 4 candidates per iteration using AVX2
        size_t d_idx = 0;
        for (; d_idx + 4 <= data_chunk_size; d_idx += 4) {
            const float* d0 = data_vectors.row(data_offset + d_idx + 0).data();
            const float* d1 = data_vectors.row(data_offset + d_idx + 1).data();
            const float* d2 = data_vectors.row(data_offset + d_idx + 2).data();
            const float* d3 = data_vectors.row(data_offset + d_idx + 3).data();
            if (d_idx + 16 < data_chunk_size) {
                __builtin_prefetch(data_vectors.row(data_offset + d_idx + 16).data(), 0, 1);
            }
            auto s = dot4_avx2(q_vec, d0, d1, d2, d3, dim);
            // HNSW InnerProduct distance is typically -dot
            float dist0 = -s[0];
            float dist1 = -s[1];
            float dist2 = -s[2];
            float dist3 = -s[3];
            if (dist0 < min_dist_for_q) min_dist_for_q = dist0;
            if (dist1 < min_dist_for_q) min_dist_for_q = dist1;
            if (dist2 < min_dist_for_q) min_dist_for_q = dist2;
            if (dist3 < min_dist_for_q) min_dist_for_q = dist3;
        }
        for (; d_idx < data_chunk_size; ++d_idx) {
            const float* d_vec = data_vectors.row(data_offset + d_idx).data();
            // Tail uses vectorized accumulation over 16 floats per iter, then one horizontal sum
            float acc = 0.0f;
            int i = 0;
            __m256 acc_ps = _mm256_setzero_ps();
            for (; i + 16 <= dim; i += 16) {
                __m256 qv0 = _mm256_loadu_ps(q_vec + i);
                __m256 dv0 = _mm256_loadu_ps(d_vec + i);
                __m256 qv1 = _mm256_loadu_ps(q_vec + i + 8);
                __m256 dv1 = _mm256_loadu_ps(d_vec + i + 8);
#if defined(__FMA__)
                acc_ps = _mm256_fmadd_ps(qv0, dv0, acc_ps);
                acc_ps = _mm256_fmadd_ps(qv1, dv1, acc_ps);
#else
                acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(qv0, dv0));
                acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(qv1, dv1));
#endif
            }
            // Handle remaining 8-wide chunk
            if (i + 8 <= dim) {
                __m256 qv = _mm256_loadu_ps(q_vec + i);
                __m256 dv = _mm256_loadu_ps(d_vec + i);
#if defined(__FMA__)
                acc_ps = _mm256_fmadd_ps(qv, dv, acc_ps);
#else
                acc_ps = _mm256_add_ps(acc_ps, _mm256_mul_ps(qv, dv));
#endif
                i += 8;
            }
            // Horizontal sum of acc_ps
            __m128 low = _mm256_castps256_ps128(acc_ps);
            __m128 high = _mm256_extractf128_ps(acc_ps, 1);
            __m128 sum = _mm_add_ps(low, high);
            __m128 shuf = _mm_movehdup_ps(sum);
            sum = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sum);
            sum = _mm_add_ss(sum, shuf);
            acc += _mm_cvtss_f32(sum);
            // Scalar tail
            for (; i < dim; ++i) acc += q_vec[i] * d_vec[i];
            float dist = -acc;
            if (dist < min_dist_for_q) min_dist_for_q = dist;
        }
#endif
        total_min_dist += min_dist_for_q;
    }
    return total_min_dist / query_chunk_size;
}

struct CompareByScore {
    bool operator()(const std::pair<float, hnswlib::labeltype>& a, 
                    const std::pair<float, hnswlib::labeltype>& b) const {
        return a.first < b.first; // Max-heap for efficient top-k maintenance
    }
};
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

    sprintf(data_file, "/data/ali/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(data_file_chunks, "/data/ali/%s/%s_base_chunks.fvecs", DATASET, DATASET);
    sprintf(query_file, "/data/ali/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(query_file_chunks, "/data/ali/%s/%s_query_chunks.fvecs", DATASET, DATASET);
    sprintf(gt_file, "/data/ali/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(result_file, "/home/ali/hnswlib/results/%s_hnsw-raw_k.csv", DATASET);
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
    std::vector<int> k_values = {1, 5, 10};
    std::vector<int> target_k_values = {1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50};
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
    // std::string index_path = "/data/ali/indexing/" + std::string(DATASET) + "_index_complete.bin";
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

    // Define Candidate structure
    struct Candidate {
        hnswlib::labeltype parent_id; // or int if you prefer
        float upper_bound;
        float lower_bound;
    };

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

    // // Query the elements for themselves and measure recall
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
    // per_query_output << "query_chunk_id,vector_index_within_chunk,k,predicted_ids" << std::endl;

    // Declare variables needed for multivector search
    float epsilon = 0.01f;  // Small epsilon for upper bound computation
    std::priority_queue<std::pair<float, hnswlib::labeltype>> top_candidates;
    ThreadPool pool(num_threads);

    // Query the k-NN for the query set and measure recall for different K
    for (int tk : target_k_values) {
        for (int k : k_values) {
            float correct_knn = 0;
            StopW stopw;
            // Loop over query DOCUMENTS (parents)
            for (int i = 0; i < NQ_chunks; i++) {
                // if (i == 1000)
                //     break;
                // Get query document info from pre-computed table
                size_t query_offset = query_parent_info[i].first;
                size_t query_chunk_size = query_parent_info[i].second;
                std::vector<std::vector<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>>> all_search_results(query_chunk_size);

                std::vector<hnswlib::labeltype> candidate_parent_ids;
                std::unordered_set<hnswlib::labeltype> candidate_parent_ids_set;
                std::unordered_map<int, std::unordered_map<int, std::vector<hnswlib::tableint>>> approx_dist_to_parent;
                std::vector<float> kth_distances(query_chunk_size);
                for (size_t j = 0; j < query_chunk_size; ++j) {
                    pool.enqueue([&, j] {
                        auto results = alg_hnsw->searchKnnSkippingDuplicates(query_vecs.row(query_offset + j).data(), tk);

                        // Store results in reverse order (since priority_queue gives largest first)
                        std::vector<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>> result_vec;
                        result_vec.reserve(results.size());
                        while (!results.empty()) {
                            result_vec.push_back(results.top());
                            results.pop();
                        }
                        // Reverse to get smallest distances first
                        std::reverse(result_vec.begin(), result_vec.end());
                        all_search_results[j] = std::move(result_vec);
                    });
                }

                // Ensure all per-vector searches are done before consuming results
                pool.wait();

                for (size_t j = 0; j < query_chunk_size; ++j) {
                    size_t query_vec_idx = query_offset + j;
                    const auto& search_results = all_search_results[j];
                    if (!search_results.empty()) {
                        size_t kth_index = std::min(static_cast<size_t>(tk - 1), search_results.size() - 1);
                        float current_kth_dist = std::get<0>(search_results[kth_index]);
                        kth_distances[j] = current_kth_dist;
                        
                        // Process all results
                        // Reserve to reduce rehashing when many parents per query
                        candidate_parent_ids_set.reserve(candidate_parent_ids_set.size() + search_results.size());
                        for (const auto& result : search_results) {
                            float approx_dist = std::get<0>(result);
                            hnswlib::labeltype parent_id = std::get<2>(result);
                            
                            // Store the best (smallest) distance for this parent-query pair
                            if (approx_dist_to_parent[parent_id][j].empty() || approx_dist < approx_dist_to_parent[parent_id][j].back()) {
                                approx_dist_to_parent[parent_id][j].push_back(approx_dist);
                            }
                            
                            candidate_parent_ids_set.insert(parent_id);
                        }
                    }
                }
                int current_k = k+20; // Or your desired initial value

                std::priority_queue<std::pair<float, hnswlib::labeltype>, 
                            std::vector<std::pair<float, hnswlib::labeltype>>,
                            CompareByScore> top_candidates;
                            
                const float epsilon = 1e-6f;

                for (auto& parent_id : candidate_parent_ids_set) {
                    float upper_bound_score = 0.0f;
                    float lower_bound_score = 0.0f;
                    float score = 0.0f;
                    for (size_t j = 0; j < query_chunk_size; ++j) {
                        if (approx_dist_to_parent[parent_id][j].empty()) {
                            upper_bound_score += kth_distances[j] + epsilon;
                            lower_bound_score += 1;
                        } else {
                            float best_dist = approx_dist_to_parent[parent_id][j].back();
                            upper_bound_score += best_dist;
                            lower_bound_score += best_dist;
                        }
                    }
                    score = upper_bound_score;
                    // Maintain top-k candidates
                    if (top_candidates.size() <= current_k) {
                        top_candidates.emplace(score, parent_id);
                    } else if (score < top_candidates.top().first) {
                        top_candidates.pop();
                        top_candidates.emplace(score, parent_id);
                    }
                }

                // Collect candidates from heap
                std::vector<std::pair<float, hnswlib::labeltype>> candidate_parents;
                candidate_parents.reserve(top_candidates.size());
                while (!top_candidates.empty()) {
                    candidate_parents.push_back(top_candidates.top());
                    top_candidates.pop();
                }

                // Parallel compute MinDist for each candidate parent
                std::vector<std::pair<float, hnswlib::labeltype>> reranked_parents(candidate_parents.size());
                for (size_t ci = 0; ci < candidate_parents.size(); ++ci) {
                    pool.enqueue([&, ci] {
                        hnswlib::labeltype parent_id = candidate_parents[ci].second;
                        size_t data_offset = data_parent_info[parent_id].first;
                        size_t data_chunk_size = data_parent_info[parent_id].second;
                        float distance = MinDist(query_vecs, data_vecs, dim,
                                                query_chunk_size, data_chunk_size,
                                                query_offset, data_offset, &space);
                        reranked_parents[ci] = {distance, parent_id};
                    });
                }
                pool.wait();

                // Sort by distance to find the best parent document
                std::sort(reranked_parents.begin(), reranked_parents.end());

                // Stage 3: Check against ground truth
                // Get the top-k predicted parent IDs into a set for fast lookup.
                std::unordered_set<hnswlib::labeltype> predicted_parent_ids;
                for (int j = 0; j < k && j < (int)reranked_parents.size(); ++j) {
                    // std::cout << "reranked_parents[" << j << "].second = " << reranked_parents[j].second << std::endl;
                    predicted_parent_ids.insert(reranked_parents[j].second);
                }

                // Get the parent ID of the single best ground truth vector.
                hnswlib::labeltype gt_vector_label = gt_vecs(i, 0);
                // hnswlib::labeltype gt_parent_id = vector_to_parent_map[gt_vector_label];

                // Check if the top ground truth parent is in our set of predicted parents.
                if (predicted_parent_ids.count(gt_vector_label)) {
                    correct_knn++;
                }

                // size_t query_offset = query_parent_info[i].first;
                // size_t query_chunk_size = query_parent_info[i].second;
                // std::vector<std::vector<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>>> all_search_results(query_chunk_size);

                // std::vector<hnswlib::labeltype> candidate_parent_ids;
                // std::unordered_set<hnswlib::labeltype> candidate_parent_ids_set;
                // std::unordered_map<int, std::unordered_map<int, std::vector<hnswlib::tableint>>> approx_dist_to_parent;
                // std::vector<float> kth_distances(query_chunk_size);
                // for (size_t j = 0; j < query_chunk_size; ++j) {
                //     pool.enqueue([&, j] {
                //         auto results = alg_hnsw->searchKnn(query_vecs.row(query_offset + j).data(), tk);

                //         // Store results in reverse order (since priority_queue gives largest first)
                //         std::vector<std::tuple<float, hnswlib::labeltype, hnswlib::labeltype>> result_vec;
                //         result_vec.reserve(results.size());
                //         while (!results.empty()) {
                //             result_vec.push_back(results.top());
                //             results.pop();
                //         }
                //         // Reverse to get smallest distances first
                //         std::reverse(result_vec.begin(), result_vec.end());
                //         all_search_results[j] = std::move(result_vec);
                //     });
                // }

                // // Ensure all per-vector searches are done before consuming results
                // pool.wait();

                // for (size_t j = 0; j < query_chunk_size; ++j) {
                //     size_t query_vec_idx = query_offset + j;
                //     const auto& search_results = all_search_results[j];
                //     if (!search_results.empty()) {
                //         candidate_parent_ids_set.reserve(candidate_parent_ids_set.size() + search_results.size());
                //         for (const auto& result : search_results) {
                //             float approx_dist = std::get<0>(result);
                //             hnswlib::labeltype parent_id = std::get<2>(result);
                //             candidate_parent_ids_set.insert(parent_id);
                //         }
                //     }
                // }
                // // Materialize candidate parent IDs into a vector for indexed access
                // candidate_parent_ids.assign(candidate_parent_ids_set.begin(), candidate_parent_ids_set.end());

                // // Parallel compute MinDist for each candidate parent
                // std::vector<std::pair<float, hnswlib::labeltype>> reranked_parents(candidate_parent_ids.size());
                // for (size_t ci = 0; ci < candidate_parent_ids.size(); ++ci) {
                //     pool.enqueue([&, ci] {
                //         hnswlib::labeltype parent_id = candidate_parent_ids[ci];
                //         size_t data_offset = data_parent_info[parent_id].first;
                //         size_t data_chunk_size = data_parent_info[parent_id].second;
                //         float distance = MinDist(query_vecs, data_vecs, dim,
                //                                 query_chunk_size, data_chunk_size,
                //                                 query_offset, data_offset, &space);
                //         reranked_parents[ci] = {distance, parent_id};
                //     });
                // }
                // pool.wait();

                // // Sort by distance to find the best parent document
                // std::sort(reranked_parents.begin(), reranked_parents.end());

                // // Stage 3: Check against ground truth
                // // Get the top-k predicted parent IDs into a set for fast lookup.
                // std::unordered_set<hnswlib::labeltype> predicted_parent_ids;
                // for (int j = 0; j < k && j < (int)reranked_parents.size(); ++j) {
                //     // std::cout << "reranked_parents[" << j << "].second = " << reranked_parents[j].second << std::endl;
                //     predicted_parent_ids.insert(reranked_parents[j].second);
                // }

                // // Get the parent ID of the single best ground truth vector.
                // hnswlib::labeltype gt_vector_label = gt_vecs(i, 0);
                // // hnswlib::labeltype gt_parent_id = vector_to_parent_map[gt_vector_label];

                // // Check if the top ground truth parent is in our set of predicted parents.
                // if (predicted_parent_ids.count(gt_vector_label)) {
                //     correct_knn++;
                // }

            }
            float total_time_us = stopw.getElapsedTimeMicro();
            float qps = NQ_chunks / (total_time_us / 1e6);
            float recall_knn = correct_knn / (float)NQ_chunks;
            // float qps = 1000 / (total_time_us / 1e6);
            // float recall_knn = correct_knn / (float)1000;
            std::cout << "k-NN Document Recall@" << k << ": " << recall_knn << ", QPS: " << qps << "\n";
            result_output << k << "," << recall_knn << "," << qps << std::endl;
        }
    }
    result_output.close();
    // delete[] data;
    delete alg_hnsw;
    return 0;
}