// g++ -O2 -std=c++17 -DS3D_BOUNDS_CHECK simple3d.hpp -o demo && ./demo

#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// Base geometry
struct Grid3D {
std::size_t NI, NJ, NK; // sizes along I, J, K (no halos)
};

// contiguous 3D array (K fastest), backed by std::vector<T>
// Access: A(i,j,k), or A[{i,j,k}]

template<typename T>
class Array3D {
public:
explicit Array3D(const Grid3D& g)
: G_(g), strideJ_(g.NK), strideI_(g.NJ * g.NK), data_(g.NI * g.NJ * g.NK) {}

// --- Element access (bounds check optional via S3D_BOUNDS_CHECK) ---
inline T& operator()(std::size_t i, std::size_t j, std::size_t k) { return data_[index(i,j,k)]; }
  inline const T& operator()(std::size_t i, std::size_t j, std::size_t k) const { return data_[index(i,j,k)]; }

// --- Utilities ---
std::size_t size_flat() const { return data_.size(); }
const Grid3D& grid() const { return G_; }
std::size_t strideJ() const { return strideJ_; } // stride when j increments by 1
std::size_t strideI() const { return strideI_; } // stride when i increments by 1
T* data() { return data_.data(); }
const T* data() const { return data_.data(); }

private:

inline std::size_t index(std::size_t i, std::size_t j, std::size_t k) const {
#ifdef S3D_BOUNDS_CHECK
if (i>=G_.NI || j>=G_.NJ || k>=G_.NK) throw std::out_of_range("Array3D: index out of range");
#endif
return i*strideI_ + j*strideJ_ + k;
}

Grid3D G_{};
std::size_t strideJ_{0}, strideI_{0}; // K fastest
std::vector<T> data_;
};
