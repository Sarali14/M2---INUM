#pragma once

#include<iostream>
#include<cmath>
#include<vector>

template <typename T>
class CS1D{
private:
  double _h;
  double _alpha,_beta;
  std::size_t _N;
  std::vector<T> _d;

public:
  CS1D(double h, std::size_t N ):
    _h(h), _N(N) {}

  ~CS1D() {}

  const double& get_step_size() const{return _h;}
  const std::size_t& get_nb_nodes() const{return _N;}
  const vector& get_derivative() {return _d;}

  

  
