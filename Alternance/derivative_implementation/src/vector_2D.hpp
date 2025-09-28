#pragma once

#include<iostream>
#include<vector>
#include<initializer_list>


struct Grid2D {
  std::size_t Ni , Nj ;
};

template <typename T>
class Array2D{
private:
  Grid2D _G{};
  std::size_t _Stride_i;
  std::vector<T> _data;
  
public:
  Array2D(const Grid2D& g) :
    _G(g), _Stride_i(g.Nj), _data(g.Ni * g.Nj) {}

  ~Array2D() {}

  inline T& operator()(std::size_t i, std::size_t j) { return _data[index(i,j)]; }
  inline const T& operator()(std::size_t i, std::size_t j) const { return _data[index(i,j)]; }
 
  std::size_t strideI() const { return _Stride_i; } 
  T* data() { return _data.data(); }
  const T* data() const { return _data.data(); }
  const Grid2D& grid() const { return _G ;}
  const std::size_t size_flat() const { return _data.size() ;}

  std::size_t index(std::size_t i, std::size_t j) const {
  #ifdef S2D_BOUNDS_CHECK
  if (i>=_Nx || j>=_Ny) throw std::out_of_range("Array2D: index out of range");
  #endif
  return i*_Stride_i + j;
  }
  
};
