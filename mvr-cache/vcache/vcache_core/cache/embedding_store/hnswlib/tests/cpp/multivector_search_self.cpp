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


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <DATASET>" << std::endl;
        return 1;
    }    int dim;               // Dimension of the elements
    int M = 256;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 256;  // Controls index search speed/build speed tradeoff
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
    sprintf(result_file, "/home/ali/hnswlib/results/%s_hnsw-self.csv", DATASET);
    sprintf(per_query_file, "/home/ali/hnswlib/results/%s_hnsw_per_query-self.csv", DATASET);

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
    
    // Test multiple thresholds from 0.09 to 0.2 with 0.01 increments
    std::vector<float> thresholds;
    // for (float t = 0.008f; t <= 0.009f; t += 0.001f) {
    //     thresholds.push_back(t);
    // }
    // thresholds.push_back(0.001f);
    // thresholds.push_back(0.002f);
    // thresholds.push_back(0.003f);
    // thresholds.push_back(0.004f);
    // thresholds.push_back(0.005f);
    // thresholds.push_back(0.006f);
    thresholds.push_back(0.007f);
    thresholds.push_back(0.008f);
    thresholds.push_back(0.009f);
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

    // std::string index_path = "/data/indexing/" + std::string(DATASET) + "_index_raw.bin";
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

    // Loop through different thresholds
    for (float threshold : thresholds) {
        std::cout << "\n=== Processing threshold: " << threshold << " ===" << std::endl;
        
        // Query the elements for themselves and collect top 15 IDs and distances
        std::vector<std::vector<hnswlib::labeltype>> top_k_neighbors(N);
        std::vector<std::vector<hnswlib::labeltype>> top_k_parents(N);
        std::vector<std::vector<float>> top_k_distances(N);
        ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {
        auto results = alg_hnsw->searchKnn(data_vecs.row(row).data(), 50);
        std::vector<hnswlib::labeltype> neighbors_for_row;
        std::vector<hnswlib::labeltype> parents_for_row;
        std::vector<float> distances_for_row;
        
        // Add the vector's own ID and parent ID with distance 0
        neighbors_for_row.push_back(row);
        parents_for_row.push_back(vector_to_parent_map[row]);
        distances_for_row.push_back(0.0f);
        
        while (!results.empty()) {
            auto t = results.top();
            results.pop();
            float distance = std::get<0>(t);                // distance
            hnswlib::labeltype label = std::get<1>(t);      // neighbor ID
            hnswlib::labeltype parent_label = std::get<2>(t); // parent ID
            // Apply threshold filter
            if (distance < threshold) {
                neighbors_for_row.push_back(label);
                parents_for_row.push_back(parent_label);
                distances_for_row.push_back(distance);
            }
        }
        // Enforce fixed length K including self; pad/truncate to k_value (written later)
        const size_t K = 16; // self + up to 15 neighbors
        if (neighbors_for_row.size() > K) {
            neighbors_for_row.resize(K);
            parents_for_row.resize(K);
            distances_for_row.resize(K);
        } else if (neighbors_for_row.size() < K) {
            // pad with -1 ids and inf distances
            while (neighbors_for_row.size() < K) {
                neighbors_for_row.push_back(static_cast<hnswlib::labeltype>(-1));
                parents_for_row.push_back(static_cast<hnswlib::labeltype>(-1));
                distances_for_row.push_back(std::numeric_limits<float>::infinity());
            }
        }
        top_k_neighbors[row] = std::move(neighbors_for_row);
        top_k_parents[row] = std::move(parents_for_row);
        top_k_distances[row] = std::move(distances_for_row);
    });

    std::vector<std::vector<int>> adj(N);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < top_k_neighbors[i].size(); j++) {
            int nb = top_k_neighbors[i][j];
            float dist = top_k_distances[i][j];
            if (nb >= 0 && dist < threshold && nb != (int)i) {
                adj[i].push_back(nb);
                adj[nb].push_back(i);
            }
        }
    }

    // ============================
    // Step 2: Cluster with BFS/DFS
    // ============================
    int max_cluster_size = 7; // your chosen cap
    std::vector<int> cluster_id(N, -1);
    int cluster_count = 0;

    hnswlib::DISTFUNC<float> dist_func = space.get_dist_func();
    void* dist_func_param = space.get_dist_func_param();

    std::vector<std::vector<int>> clusters;

    // --- BFS-based strict clustering ---
    for (int i = 0; i < N; i++) {
        if (cluster_id[i] != -1) continue; // already clustered

        std::queue<int> q;
        std::vector<int> current_cluster;
        q.push(i);
        cluster_id[i] = cluster_count;
        current_cluster.push_back(i);

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int v : adj[u]) {
                if (cluster_id[v] != -1) continue;
                if (current_cluster.size() >= (size_t)max_cluster_size) continue;

                bool all_close = true;
                for (int m : current_cluster) {
                    float d = dist_func(
                        data_vecs.row(v).data(),
                        data_vecs.row(m).data(),
                        dist_func_param
                    );
                    if (d >= threshold) {
                        all_close = false;
                        break;
                    }
                }

                if (all_close) {
                    cluster_id[v] = cluster_count;
                    q.push(v);
                    current_cluster.push_back(v);
                }
            }
        }

        clusters.push_back(std::move(current_cluster));
        cluster_count++;
    }

std::cout << "Formed " << cluster_count << " strict clusters" << std::endl;

    struct Representative {
        int rep_vec_id;                        // representative vector ID
        std::vector<int> parent_ids;           // all parents in cluster
        std::vector<int> member_vec_ids;       // all vectors in cluster
    };


    std::vector<Representative> representatives(cluster_count);

    // Step 3a: Pick representative & gather member vector IDs
    std::vector<int> rep_vec(cluster_count, -1);
    for (int i = 0; i < N; i++) {
        int cid = cluster_id[i];

        if (rep_vec[cid] == -1) rep_vec[cid] = i; // pick first vector as representative

        representatives[cid].member_vec_ids.push_back(i);
    }

    // Step 3b: Aggregate parent IDs per cluster
    std::vector<std::unordered_set<int>> cluster_parents(cluster_count);
    for (int i = 0; i < N; i++) {
        int cid = cluster_id[i];                // cluster that vector i belongs to
        int pid = vector_to_parent_map[i];      // parent of vector i itself
        cluster_parents[cid].insert(pid);       // add to this cluster’s parent set
    }

    for (int cid = 0; cid < cluster_count; cid++) {
        representatives[cid].rep_vec_id = rep_vec[cid];
        representatives[cid].parent_ids = std::vector<int>(
            cluster_parents[cid].begin(), cluster_parents[cid].end()
        );
    }
    std::cout << "Representatives selected" << std::endl;

    // Compute average (centroid) vector per cluster to use as representative
    std::vector<std::vector<float>> rep_means(cluster_count, std::vector<float>(dim, 0.0f));
    for (int cid = 0; cid < cluster_count; cid++) {
        const std::vector<int> &members = representatives[cid].member_vec_ids;
        if (members.empty()) continue;
        // Sum member vectors
        for (int vid : members) {
            const float* v = data_vecs.row(vid).data();
            for (int d = 0; d < dim; d++) {
                rep_means[cid][d] += v[d];
            }
        }
        // Divide by count to get mean
        const float inv_count = 1.0f / static_cast<float>(members.size());
        for (int d = 0; d < dim; d++) {
            rep_means[cid][d] *= inv_count;
        }
    }

        // Save representative vectors (cluster means)
        char vec_file_path[500];
        sprintf(vec_file_path, "/data/ali/clustering-clef/rep_vectors_thresh_%.3f.fvecs", threshold);
        std::ofstream vec_file(vec_file_path, std::ios::binary);
    for (int cid = 0; cid < cluster_count; cid++) {
        int d = dim;
        vec_file.write((char*)&d, sizeof(int)); // optional: store dim
        vec_file.write((char*)rep_means[cid].data(), sizeof(float)*dim);
    }
    vec_file.close();

        // Save parent IDs
        char parent_file_path[500];
        sprintf(parent_file_path, "/data/ali/clustering-clef/rep_chunks_thresh_%.3f.ivecs", threshold);
        std::ofstream parent_file(parent_file_path, std::ios::binary);
    for (const auto& rep : representatives) {
        int n = rep.parent_ids.size();  // number of parent IDs
        parent_file.write((char*)&n, sizeof(int));
        parent_file.write((char*)rep.parent_ids.data(), sizeof(int) * n);
    }
    parent_file.close();

        // Save member vector IDs
        char member_file_path[500];
        sprintf(member_file_path, "/data/ali/clustering-clef/member_vectors_thresh_%.3f.fvecs", threshold);
        std::ofstream member_file(member_file_path, std::ios::binary);
    for (const auto& rep : representatives) {
        int n = rep.member_vec_ids.size();  // number of member vector IDs
        member_file.write((char*)&n, sizeof(int));
        member_file.write((char*)rep.member_vec_ids.data(), sizeof(int) * n);
    }
    member_file.close();


    
        // Save top 15 neighbor IDs to binary file
        char neighbors_file[500];
        sprintf(neighbors_file, "/data/ali/clustering-clef/%s_neighbors_thresh_%.3f.bin", DATASET, threshold);
    std::ofstream neighbors_outfile(neighbors_file, std::ios::binary);
    if (!neighbors_outfile.is_open()) {
        std::cerr << "Error: Could not open neighbors output file " << neighbors_file << std::endl;
        return 1;
    }
    
    // Write number of vectors and k value to neighbors file
    uint32_t num_vectors = N;
    uint32_t k_value = 16;
    neighbors_outfile.write(reinterpret_cast<const char*>(&num_vectors), sizeof(uint32_t));
    neighbors_outfile.write(reinterpret_cast<const char*>(&k_value), sizeof(uint32_t));
    
    // Write the neighbor IDs for each vector
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < top_k_neighbors[i].size(); ++j) {
            int32_t neighbor_id = static_cast<int32_t>(top_k_neighbors[i][j]);
            neighbors_outfile.write(reinterpret_cast<const char*>(&neighbor_id), sizeof(int32_t));
        }
    }
    neighbors_outfile.close();
    std::cout << "Top 15 neighbors saved to: " << neighbors_file << std::endl;
    
        // Save top 15 parent IDs to separate binary file
        char parents_file[500];
        sprintf(parents_file, "/data/ali/clustering-clef/%s_parents_thresh_%.3f.bin", DATASET, threshold);
    std::ofstream parents_outfile(parents_file, std::ios::binary);
    if (!parents_outfile.is_open()) {
        std::cerr << "Error: Could not open parents output file " << parents_file << std::endl;
        return 1;
    }
    
    // Write number of vectors and k value to parents file
    parents_outfile.write(reinterpret_cast<const char*>(&num_vectors), sizeof(uint32_t));
    parents_outfile.write(reinterpret_cast<const char*>(&k_value), sizeof(uint32_t));
    
    // Write the parent IDs for each vector
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < top_k_parents[i].size(); ++j) {
            int32_t parent_id = static_cast<int32_t>(top_k_parents[i][j]);
            parents_outfile.write(reinterpret_cast<const char*>(&parent_id), sizeof(int32_t));
        }
    }
    parents_outfile.close();
    std::cout << "Top 15 parents saved to: " << parents_file << std::endl;
    
        // Save top 15 distances to separate binary file
        char distances_file[500];
        sprintf(distances_file, "/data/ali/clustering-clef/%s_distances_thresh_%.3f.bin", DATASET, threshold);
    std::ofstream distances_outfile(distances_file, std::ios::binary);
    if (!distances_outfile.is_open()) {
        std::cerr << "Error: Could not open distances output file " << distances_file << std::endl;
        return 1;
    }
    
    // Write number of vectors and k value to distances file
    distances_outfile.write(reinterpret_cast<const char*>(&num_vectors), sizeof(uint32_t));
    distances_outfile.write(reinterpret_cast<const char*>(&k_value), sizeof(uint32_t));
    
    // Write the distances for each vector
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < top_k_distances[i].size(); ++j) {
            float distance = top_k_distances[i][j];
            distances_outfile.write(reinterpret_cast<const char*>(&distance), sizeof(float));
        }
    }
    distances_outfile.close();
    std::cout << "Top 15 distances saved to: " << distances_file << std::endl;
    
        // Save cluster id for each point in ivecs format (length=1, then cid)
        char cluster_file[500];
        sprintf(cluster_file, "/data/ali/clustering-clef/%s_cluster_ids_thresh_%.3f.ivecs", DATASET, threshold);
    std::ofstream cluster_outfile(cluster_file, std::ios::binary);
    if (!cluster_outfile.is_open()) {
        std::cerr << "Error: Could not open cluster ids output file " << cluster_file << std::endl;
        return 1;
    }
    // Write one ivec per point: [length=1][cid]
    for (size_t i = 0; i < N; ++i) {
        int32_t len = 1;
        int32_t cid = static_cast<int32_t>(cluster_id[i]);
        cluster_outfile.write(reinterpret_cast<const char*>(&len), sizeof(int32_t));
        cluster_outfile.write(reinterpret_cast<const char*>(&cid), sizeof(int32_t));
    }
    cluster_outfile.close();
    std::cout << "Cluster IDs saved to: " << cluster_file << std::endl;
    
        // Calculate self-recall (check if the vector itself exists in top 15 neighbors)
        float correct = 0;
        for (int i = 0; i < N; i++) {
            bool found_self = false;
            for (size_t j = 0; j < top_k_neighbors[i].size(); ++j) {
                if (top_k_neighbors[i][j] == i) {
                    found_self = true;
                    break;
                }
            }
            if (found_self) {
                correct++;
            }
        }
        float recall = correct / (float)N;
        std::cout << "Self-Recall for threshold " << threshold << ": " << recall << std::endl;
        std::cout << "=== Completed threshold: " << threshold << " ===" << std::endl;
    }
    // delete[] data;
    delete alg_hnsw;
    return 0;
}