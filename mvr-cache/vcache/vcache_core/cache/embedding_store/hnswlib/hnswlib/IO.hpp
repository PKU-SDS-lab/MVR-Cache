#pragma once

#include <stdint.h>
#include <sys/stat.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

size_t get_filesize(const char* filename) {
    struct stat64 stat_buf;
    int rc = stat64(filename, &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

bool file_exits(const char* filename) {
    std::ifstream f(filename);
    if (!f.good()) {
        f.close();
        return false;
    }
    f.close();
    return true;
}

template <typename T>
T* load_vecs(const char* filename) {
    if (!file_exits(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    uint32_t cols;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read((char*)&cols, sizeof(uint32_t));

    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    T* data = new T[rows * cols];

    input.seekg(0, input.beg);

    for (size_t i = 0; i < rows; i++) {
        input.read((char*)&cols, sizeof(uint32_t));
        input.read((char*)&data[cols * i], sizeof(T) * cols);
    }

    input.close();
    return data;
}

template <typename T, class M>
void load_vecs(const char* filename, M& Mat) {
    if (!file_exits(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    T* ptr;
    assert(typeid(ptr) == typeid(Mat.data()));

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read((char*)&tmp, sizeof(uint32_t));

    size_t cols = tmp;
    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    Mat = M(rows, cols);

    input.seekg(0, input.beg);

    for (size_t i = 0; i < rows; i++) {
        input.read((char*)&tmp, sizeof(uint32_t));
        input.read((char*)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}

std::vector<std::vector<std::vector<float>>>
load_fvecs_3d(const std::string& filename)
{
    std::ifstream in{filename, std::ios::binary};
    if (!in) {
        throw std::runtime_error("Cannot open " + filename);
    }

    // total file size
    in.seekg(0, std::ios::end);
    std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // peek first row header
    int32_t C = 0, DIM = 0;
    in.read(reinterpret_cast<char*>(&C),   sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&DIM), sizeof(int32_t));
    if (!in) {
        throw std::runtime_error("Failed to read header from " + filename);
    }

    // bytes per row = 4 + C*(4 + DIM*4)
    const std::size_t bytes_per_row =
        sizeof(int32_t)
      + std::size_t(C) * ( sizeof(int32_t) + std::size_t(DIM)*sizeof(float) );

    if (file_size % bytes_per_row != 0) {
        throw std::runtime_error("Corrupt file: size mismatch");
    }

    const std::size_t N = file_size / bytes_per_row;
    std::vector<std::vector<std::vector<float>>> data(
        N,
        std::vector<std::vector<float>>(C, std::vector<float>(DIM))
    );

    // rewind and read every row
    in.clear();
    in.seekg(0, std::ios::beg);
    for (std::size_t i = 0; i < N; ++i) {
        // read and discard the per-row C
        in.read(reinterpret_cast<char*>(&C), sizeof(int32_t));

        for (int c = 0; c < C; ++c) {
            // read chunk length (should equal DIM)
            in.read(reinterpret_cast<char*>(&DIM), sizeof(int32_t));
            assert(DIM == static_cast<int32_t>(data[i][c].size()) &&
                   "Chunk size mismatch!");

            // read DIM floats into data[i][c][0]
            in.read(reinterpret_cast<char*>(data[i][c].data()),
                    DIM * sizeof(float));
        }
    }

    std::cout << "Loaded tensor “" << filename << "” "
              << "with N=" << N
              << ", C=" << data[0].size()
              << ", DIM=" << data[0][0].size()
              << "\n";

    return data;
}


template <typename T, class M>
void load_bin(const char* filename, M& Mat) {
    if (!file_exits(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    T* ptr;
    assert(typeid(ptr) == typeid(Mat.data()));

    uint32_t rows, cols;
    std::ifstream input(filename, std::ios::binary);

    input.read((char*)&rows, sizeof(uint32_t));
    input.read((char*)&cols, sizeof(uint32_t));

    Mat = M(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        input.read((char*)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}