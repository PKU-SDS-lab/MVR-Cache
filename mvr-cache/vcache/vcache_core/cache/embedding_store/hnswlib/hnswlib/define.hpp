#pragma once

#include <stdint.h>
#include <vector>

#include "third/Eigen/Dense"

#define FORCE_INLINE inline __attribute__((always_inline))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define lowbit(x) (x & (-x))
#define bit_id(x) (__builtin_popcount(x - 1))

using PID = uint32_t;
using pair_di = std::pair<double, int>;
using FloatRowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntRowMat = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using UintRowMat = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DoubleRowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;