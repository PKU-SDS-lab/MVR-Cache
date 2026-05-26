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
    int num_threads = 10;       // Number of threads for operations with index
    const char* DATASET = argv[1];

    int K = 4096;

    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char result_file[500];
    char centroids_file[500];
    char cids_file[500];
    char data_file_chunks[500];
    char query_file_chunks[500];

    sprintf(data_file, "/data1/ali/data/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(data_file_chunks, "/data1/ali/data/%s/%s_base_chunks.fvecs", DATASET, DATASET);
    sprintf(query_file, "/data1/ali/data/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(query_file_chunks, "/data1/ali/data/%s/%s_query_chunks.fvecs", DATASET, DATASET);
    sprintf(gt_file, "/data1/ali/data/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(centroids_file, "/data1/ali/data/%s/%s_centroid_%ld.fvecs", DATASET, DATASET, K);
    sprintf(cids_file, "/data1/ali/data/%s/%s_cluster_id_%ld.ivecs", DATASET, DATASET, K);
    // sprintf(gt_file, "/data1/ali/data/%s/%s_groundtruth.ivecs");
    sprintf(result_file, "/data2/ali/similarities_cluster/%s_hnsw.csv", DATASET);

	FloatRowMat data_vecs;
    FloatRowMat data_chunks;
    FloatRowMat query_vecs;
    FloatRowMat query_chunks;
    UintRowMat gt_vecs;
    FloatRowMat centroids;
    UintRowMat cids;

    load_vecs<float, FloatRowMat>(data_file, data_vecs);
    load_vecs<float, FloatRowMat>(data_file_chunks, data_chunks);
    load_vecs<float, FloatRowMat>(query_file, query_vecs);
    load_vecs<float, FloatRowMat>(query_file_chunks, query_chunks);
    load_vecs<PID, UintRowMat>(gt_file, gt_vecs);
    load_vecs<float, FloatRowMat>(centroids_file, centroids);
    load_vecs<PID, UintRowMat>(cids_file, cids);


    size_t N = data_vecs.rows();
    dim = data_vecs.cols();
    size_t NQ = query_vecs.rows();
    size_t NQ_chunks = query_chunks.rows();

	std::cout << "data loaded\n";
    std::cout << "\tN: " << N <<  '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';

    hnswlib::InnerProductSpace space(dim);
    hnswlib::DISTFUNC<float> dist_func = space.get_dist_func();
    void* dist_func_param = space.get_dist_func_param();



    // Binary format: much more space efficient
    std::string binary_result_file = std::string(result_file);
    binary_result_file.replace(binary_result_file.find(".csv"), 4, ".bin");
    
    std::string binary_cluster_file = std::string(result_file);
    binary_cluster_file.replace(binary_cluster_file.find(".csv"), 4, "_clusters.bin");

    std::string binary_cluster_distances_file = std::string(result_file);
    binary_cluster_distances_file.replace(binary_cluster_distances_file.find(".csv"), 4, "_clusters_distances.bin");

    std::string binary_cluster_ids_file = std::string(result_file);
    binary_cluster_ids_file.replace(binary_cluster_ids_file.find(".csv"), 4, "_clusters_ids.bin");
    
    std::ofstream result_output(binary_result_file, std::ios::binary);
    std::ofstream cluster_output(binary_cluster_file, std::ios::binary);
    std::ofstream cluster_distances_output(binary_cluster_distances_file, std::ios::binary);
    std::ofstream cluster_ids_output(binary_cluster_ids_file, std::ios::binary);

    if (!result_output.is_open()) {
        std::cerr << "Failed to open result file: " << binary_result_file << std::endl;
        return 1;
    }
    
    if (!cluster_distances_output.is_open()) {
        std::cerr << "Failed to open cluster distances file: " << binary_cluster_distances_file << std::endl;
        return 1;
    }
    
    if (!cluster_output.is_open()) {
        std::cerr << "Failed to open cluster file: " << binary_cluster_file << std::endl;
        return 1;
    }

    if (!cluster_ids_output.is_open()) {
        std::cerr << "Failed to open cluster ids file: " << binary_cluster_ids_file << std::endl;
        return 1;
    }
    
    // Write headers to both files: NQ, N (as uint32_t)
    // Distances file: NQ * N floats
    // Clusters file: NQ * N uint32_t
    uint32_t nq_uint = static_cast<uint32_t>(1000);
    uint32_t n_uint = static_cast<uint32_t>(N);
    uint32_t k_cluster = static_cast<uint32_t>(K);
    
    // Write headers to both files
    result_output.write(reinterpret_cast<const char*>(&nq_uint), sizeof(uint32_t));
    result_output.write(reinterpret_cast<const char*>(&n_uint), sizeof(uint32_t));
    
    cluster_output.write(reinterpret_cast<const char*>(&nq_uint), sizeof(uint32_t));
    cluster_output.write(reinterpret_cast<const char*>(&n_uint), sizeof(uint32_t));
    
    cluster_distances_output.write(reinterpret_cast<const char*>(&nq_uint), sizeof(uint32_t));
    cluster_distances_output.write(reinterpret_cast<const char*>(&k_cluster), sizeof(uint32_t));

    cluster_ids_output.write(reinterpret_cast<const char*>(&nq_uint), sizeof(uint32_t));
    cluster_ids_output.write(reinterpret_cast<const char*>(&k_cluster), sizeof(uint32_t));
    
    // Process one query at a time to reduce memory usage from O(NQ*N) to O(N)
    for (int query_row = 0; query_row < 1000; ++query_row) {
        std::vector<float> query_distances_cluster(K);
        std::vector<uint32_t> query_cluster_ids_cluster(K);
        ParallelFor(0, K, num_threads, [&](size_t data_row, size_t threadId) {
            float dist = dist_func(query_vecs.row(query_row).data(), centroids.row(data_row).data(), dist_func_param);
            query_distances_cluster[data_row] = dist;
            query_cluster_ids_cluster[data_row] = data_row;
        });
        cluster_distances_output.write(reinterpret_cast<const char*>(query_distances_cluster.data()), query_distances_cluster.size() * sizeof(float));
        cluster_ids_output.write(reinterpret_cast<const char*>(query_cluster_ids_cluster.data()), query_cluster_ids_cluster.size() * sizeof(uint32_t));
    }


    for (int query_row = 0; query_row < 1000; ++query_row) {
        
        // Calculate similarities for all data points and store with cluster IDs
        std::vector<float> query_distances(N);
        std::vector<uint32_t> query_cluster_ids(N);
        
        ParallelFor(0, N, num_threads, [&](size_t data_row, size_t threadId) {
            float dist = dist_func(query_vecs.row(query_row).data(), data_vecs.row(data_row).data(), dist_func_param);
            size_t cluster_id = cids(data_row, 0); // Assuming each data point belongs to one cluster
            query_distances[data_row] = dist;
            query_cluster_ids[data_row] = static_cast<uint32_t>(cluster_id);
        });
        
        // Write distances for this query to distances file (N floats)
        for (size_t i = 0; i < N; ++i) {
            result_output.write(reinterpret_cast<const char*>(&query_distances[i]), sizeof(float));
            cluster_output.write(reinterpret_cast<const char*>(&query_cluster_ids[i]), sizeof(uint32_t));
        }
        
        // Optional: Print progress for large datasets
        if ((query_row + 1) % 100 == 0) {
            std::cout << "Processed " << (query_row + 1) << "/1000 queries" << std::endl;
        }
    }
    
    result_output.close();
    cluster_output.close();
    cluster_distances_output.close();
    std::cout << "Binary distances saved to: " << binary_result_file << std::endl;
    std::cout << "Binary clusters saved to: " << binary_cluster_file << std::endl;
    std::cout << "Binary cluster distances saved to: " << binary_cluster_distances_file << std::endl;
    return 0;
}
