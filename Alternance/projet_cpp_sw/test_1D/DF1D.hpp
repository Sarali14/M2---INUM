#pragma once

#include<iostream>
#include<cmath>
#include<vector>

template <typename T>
class DF1D{
  
protected:
  double _h;
  std::size_t _N;
  std::vector<T> _f;
  std::vector<T> _x;
  std::vector<T> _d;

public:
  DF1D(double h, std::size_t N)
    :_h(h), _N(N), _f(N), _x(N), _d(N) {}
  
  virtual ~DF1D() {}
  
  void load_x(){
         for (size_t i = 0; i < _N; ++i) {
       _x[i] = i * _h;
     }
  }
  
  const std::vector<T>& get_x() const { return _x; }
  
  template <typename Func>
  void load_f(Func f){
    for(std::size_t i=0; i< _x.size() ;++i){
      _f[i]= f(_x[i]);
    }}
  
  const std::vector<T>& get_function() const { return _f; }
  const std::size_t& get_nb_points() const { return _N; }
  const double& get_step_h() const { return _h; }
  const std::vector<T>& get_derivative() { return _d; }

  virtual void first_der() = 0;
  virtual void second_der() =0;
};

template <typename T>
class DF1D_o2 : public DF1D<T>{
public:
  using DF1D<T>::_N;
  using DF1D<T>::_h;
  using DF1D<T>::_f;
  using DF1D<T>::_d;

  DF1D_o2(double h, std::size_t N) : DF1D<T>(h, N) {}
  
  void first_der() override {
    for (std::size_t i=0; i< _N ; ++i){
     size_t ip1 = (i + 1) % _N;
     size_t im1 = (i + _N - 1) % _N; 
     _d[i] = (_f[ip1] - _f[im1]) / (2.0 * _h);
    }
  }
  void second_der() override {
    for (std::size_t i=0; i< _N ; ++i){
     size_t ip1 = (i + 1) % _N;
     size_t im1 = (i + _N - 1) % _N;
     _d[i] = (_f[ip1] - 2.0*_f[i] + _f[im1]) / std::pow(_h,2.0);
    }
  }

};

template <typename T>
class DF1D_o4 : public DF1D<T>{
public:
  using DF1D<T>::_N;
  using DF1D<T>::_h;
  using DF1D<T>::_f;
  using DF1D<T>::_d;

  DF1D_o4(double h, std::size_t N) : DF1D<T>(h, N) {}
  
  void first_der() override {
    for	(std::size_t i=0; i< _N ; ++i){
      std::size_t ip1 = (i+1)%_N;
      std::size_t ip2 = (i+2)%_N;
      std::size_t im1 = (i+_N-1)%_N;
      std::size_t im2 = (i+_N-2)%_N;
      
      _d[i] = (-_f[ip2] + 8.0*_f[ip1] - 8.0*_f[im1] + _f[im2]) / (12.0 * _h);
    }
  }
  void second_der() override {
    for (std::size_t i=0; i< _N ; ++i){
      std::size_t ip1 = (i+1)%_N;
      std::size_t ip2 = (i+2)%_N;
      std::size_t im1 = (i+_N-1)%_N;
      std::size_t im2 = (i+_N-2)%_N;

      _d[i] = (-_f[ip2] + 16.0*_f[ip1] - 30.0*_f[i] + 16.0*_f[im1] - _f[im2]) / (12.0 * std::pow(_h,2.0));
    }
  }
};
