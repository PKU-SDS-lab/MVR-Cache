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
#include <iostream>
#include <string>
#include <stdexcept>


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


int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <DATASET> <EF>" << std::endl;
        return 1;
    }
    int dim;               // Dimension of the elements
    int M = 256;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 50;       // Number of threads for operations with index
    const char* DATASET = argv[1];
    float ef = std::atof(argv[2]);
    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char result_file[500];
    char data_file_chunks[500];
    char query_file_chunks[500];
    // float target_k_multiplier = std::atof(argv[2]);

    sprintf(data_file, "/data/ali/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(query_file, "/data/ali/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(gt_file, "/data/ali/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(result_file, "/home/ali/hnswlib/results/%s_hnsw-single.csv", DATASET);

    FloatRowMat data_vecs;
    FloatRowMat query_vecs;
    UintRowMat gt_vecs;

    load_vecs<float, FloatRowMat>(data_file, data_vecs);
    load_vecs<float, FloatRowMat>(query_file, query_vecs);
    load_vecs<PID, UintRowMat>(gt_file, gt_vecs);

    size_t N = data_vecs.rows();
    dim = data_vecs.cols();
    size_t NQ = query_vecs.rows();
    std::vector<int> k_values = {10};
    std::vector<float> target_k_multipliers = {1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50};
    // std::vector<float> target_k_multipliers = {1.0f};
    int max_k = *std::max_element(k_values.begin(), k_values.end());
    
    // Variables for target k calculation
    int target_k;

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

    // std::string index_path = "/data2/ali/data/" + std::string(DATASET) + "_index_single.bin";
    // std::cout << "Loading index from: " << index_path << std::endl;
    
    // StopW stopw_loading;
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    // float loading_time_s = stopw_loading.getElapsedTimeMicro() / 1e6;
    // std::cout << "Index loading time: " << loading_time_s << " s" << std::endl;
    // std::cout << "Index loaded successfully" << std::endl;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, N, M, ef_construction);


    // Add data to index
    StopW stopw_indexing;
    ParallelFor(0, N, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data_vecs.row(row).data()), row, row);
    });
    float indexing_time_s = stopw_indexing.getElapsedTimeMicro() / 1e6;
    std::cout << "Indexing time: " << indexing_time_s << " s" << std::endl;

    // std::string index_path = "/data2/ali/data/" + std::string(DATASET) + "_index_single.bin";
    // std::cout << "Saving index to: " << index_path << std::endl;
    // alg_hnsw->saveIndex(index_path);
    // std::cout << "Index saved successfully" << std::endl;

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
    result_output << "k,Recall,QPS,ef" << std::endl;

    // Query the k-NN for the query set and measure recall for different K
    for (float target_k_multiplier : target_k_multipliers) {
        for (int k : k_values) {
            float correct_knn = 0;
            StopW stopw;
            for (int i = 0; i < NQ; i++) {
                if (i == 1000) break;
                auto result = alg_hnsw->searchKnnEfficient(query_vecs.row(i).data(), k, target_k_multiplier);
                
                // The ground truth is the first neighbor from the loaded file
                hnswlib::labeltype ground_truth_neighbor = gt_vecs(i, 0);

                while(result.size()){
                    auto n = result.top();
                    result.pop();
                    
                    float distance = std::get<0>(n);
                    hnswlib::labeltype label = std::get<1>(n);
                    hnswlib::labeltype parent_label = std::get<2>(n);

                    // std::cout << "Query " << i << ", Neighbor " << label << ", Parent " << parent_label << ", Distance " << distance << std::endl;

                    // std::vector<hnswlib::labeltype> samples = alg_hnsw->getSamplesByParentId(parent_label);
                    // std::cout << "  Samples with parent ID " << parent_label << ": ";
                    // for (size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx) {
                    //     std::cout << samples[sample_idx] << (sample_idx == samples.size() - 1 ? "" : ", ");
                    // }
                    // std::cout << std::endl;

                    if(label == ground_truth_neighbor){
                        correct_knn++;
                        break; // Found the match, no need to check further for this query
                    }
                }
            }
            float total_time_us = stopw.getElapsedTimeMicro();
            float qps = 1000 / (total_time_us / 1e6);
            float recall_knn = correct_knn / (float)1000;
            std::cout << "k-NN Recall@" << k << ": " << recall_knn << ", QPS: " << qps << "\n";
            result_output << k << "," << recall_knn << "," << qps << "," << target_k_multiplier << std::endl;
        }
    }

    result_output.close();

    // delete[] data;
    delete alg_hnsw;
    return 0;
}
