#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <fftw3.h>
#include "vector_3D.hpp"

struct domain_size{
  double Lx{}, Ly{} , Lz{};
};

struct step_size{
  double hi, hj , hk;
  step_size(const Grid3D& g, const domain_size& L):
    hi(L.Lx/static_cast<double>(g.NI)) ,
    hj(L.Ly/static_cast<double>(g.NJ)) ,
    hk(L.Lz/static_cast<double>(g.NK)) {}
  
};

template <typename T, typename Derived>
class derivative_3D{
protected:
  step_size _h{};
  Grid3D _G{};
  Array3D<T> _dx , _dy ,_dz , _delta;
  bool _dx_computed = false;
  bool _dy_computed = false;
  bool _dz_computed = false;

public:
  derivative_3D(const step_size& h, const Grid3D& g):
    _h(h),
    _G(g),
    _dx(g),
    _dy(g),
    _dz(g),
    _delta(g) {}

  ~derivative_3D() {}

  const Grid3D& get_nb_nodes() const{ return _G;}
  const step_size& get_step_size() const { return _h;}
  const Array3D<T>& get_derivative_x() { return _dx;}
  const Array3D<T>& get_derivative_y() { return _dy;}
  const Array3D<T>& get_derivative_z() { return _dz;}
  const Array3D<T>& get_laplacien() { return _delta;}

  template <typename Func>
    void second_der_x(Func f) {
        static_cast<Derived*>(this)->second_der_x_impl(f);
        _dx_computed = true;
    }

    template <typename Func>
    void second_der_y(Func f) {
        static_cast<Derived*>(this)->second_der_y_impl(f);
        _dy_computed = true;
    }

    template <typename Func>
    void second_der_z(Func f) {
        static_cast<Derived*>(this)->second_der_z_impl(f);
        _dz_computed = true;
    }

  template <typename Func>
  void laplacien(Func f){
    if(!_dx_computed) second_der_x(f);
    if(!_dy_computed) second_der_y(f);
    if(!_dz_computed) second_der_z(f);

    for (std::size_t i=0; i<_G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
        for (std::size_t k=0; k<_G.NK; ++k){
          _delta(i,j,k) = _dx(i,j,k)+_dy(i,j,k)+_dz(i,j,k);
        }
      }
    }
  }
};

template <typename T>
class DF_o2 : public derivative_3D<T, DF_o2<T>>{
public:
  using derivative_3D<T ,DF_o2<T> >::_h;
  using derivative_3D<T,DF_o2<T> >::_G;
  using derivative_3D<T,DF_o2<T>>::_dx;
  using derivative_3D<T,DF_o2<T>>::_dy;
  using derivative_3D<T,DF_o2<T>>::_dz;
  using derivative_3D<T,DF_o2<T>>::_delta;
  using derivative_3D<T,DF_o2<T>>::_dx_computed;
  using derivative_3D<T,DF_o2<T>>::_dy_computed;
  using derivative_3D<T,DF_o2<T>>::_dz_computed;
  
  DF_o2(const step_size& h , const Grid3D& g) : derivative_3D<T,DF_o2<T>>(h, g) {}

  template <typename Func>
  void second_der_x_impl(Func f) {
    for (std::size_t i=0; i< _G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
	for (std::size_t k=0; k<_G.NK; ++k){
	std::size_t ip1 = (i + 1) % _G.NI;
	std::size_t im1 = (i + _G.NI - 1) % _G.NI;
	_dx(i,j,k) = (f(ip1,j,k) - 2.0*f(i,j,k) + f(im1,j,k)) / std::pow(_h.hi,2.0);
	}
      }
    }
    _dx_computed = true;
  }
  template <typename Func>
  void second_der_y_impl(Func f) {
    for (std::size_t i=0; i< _G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
	for(std::size_t k=0; k<_G.NK; ++k){
	std::size_t jp1 = (j + 1) % _G.NJ;
	std::size_t jm1 = (j + _G.NJ - 1) % _G.NJ;
        _dy(i,j,k) = (f(i,jp1,k) - 2.0*f(i,j,k) + f(i,jm1,k)) / std::pow(_h.hj,2.0);
	}
      }
    }
    _dy_computed = true;
  }
  template <typename Func>
  void second_der_z_impl(Func f) {
    for (std::size_t i=0; i< _G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
        for(std::size_t k=0; k<_G.NK; ++k){
        std::size_t kp1 = (k + 1) % _G.NK;
        std::size_t km1 = (k + _G.NK - 1) % _G.NK;
        _dz(i,j,k) = (f(i,j,kp1) - 2.0*f(i,j,k) + f(i,j,km1)) / std::pow(_h.hk,2.0);
        }
      }
    }
    _dz_computed = true;
  }
};

template <typename T>
class DF_o4 : public derivative_3D<T,DF_o4<T>>{
public:
  using derivative_3D<T,DF_o4<T>>::_h;
  using derivative_3D<T,DF_o4<T>>::_G;
  using derivative_3D<T,DF_o4<T>>::_dx;
  using derivative_3D<T,DF_o4<T>>::_dy;
  using derivative_3D<T,DF_o4<T>>::_dz;
  using derivative_3D<T,DF_o4<T>>::_delta;
  using derivative_3D<T,DF_o4<T>>::_dx_computed;
  using derivative_3D<T,DF_o4<T>>::_dy_computed;
  using derivative_3D<T,DF_o4<T>>::_dz_computed;
  
  DF_o4(const step_size& h , const Grid3D& g) : derivative_3D<T,DF_o4<T>>(h, g) {}

  template <typename Func>
  void second_der_x_impl(Func f)  {
    for (std::size_t i=0; i<_G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
	for (std::size_t k=0; k<_G.NK; ++k){
	std::size_t ip1 = (i+1)%_G.NI;
	std::size_t ip2 = (i+2)%_G.NI;
	std::size_t im1 = (i+_G.NI-1)%_G.NI;
	std::size_t im2 = (i+_G.NI-2)%_G.NI;

	_dx(i,j,k) = (-f(ip2,j,k) + 16.0*f(ip1,j,k) - 30.0*f(i,j,k)
		      + 16.0*f(im1,j,k)- f(im2,j,k))/(12.0 * std::pow(_h.hi,2.0));
	}
      }
    }
    _dx_computed = true;
  }
  template <typename Func>
  void second_der_y_impl(Func f)  {
    for (std::size_t i=0; i<_G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
        for (std::size_t k=0; k<_G.NK; ++k){
        std::size_t jp1 = (j+1)%_G.NJ;
        std::size_t jp2 = (j+2)%_G.NJ;
        std::size_t jm1 = (j+_G.NJ-1)%_G.NJ;
        std::size_t jm2 = (j+_G.NJ-2)%_G.NJ;

        _dy(i,j,k) = (-f(i,jp2,k) + 16.0*f(i,jp1,k) - 30.0*f(i,j,k)
                      + 16.0*f(i,jm1,k)- f(i,jm2,k))/(12.0 * std::pow(_h.hj,2.0));
        }
      }
    }
    _dy_computed = true;
  }
  template <typename Func>
  void second_der_z_impl(Func f) {
    for (std::size_t i=0; i<_G.NI ; ++i){
      for (std::size_t j=0; j<_G.NJ; ++j){
        for (std::size_t k=0; k<_G.NK; ++k){
	std::size_t kp1 = (k+1)%_G.NK;
        std::size_t kp2 = (k+2)%_G.NK;
        std::size_t km1 = (k+_G.NK-1)%_G.NK;
        std::size_t km2 = (k+_G.NK-2)%_G.NK;

        _dz(i,j,k) = (-f(i,j,kp2) + 16.0*f(i,j,kp1) - 30.0*f(i,j,k)
                      + 16.0*f(i,j,km1)- f(i,j,km2))/(12.0 * std::pow(_h.hk,2.0));
        }
      }
    }
    _dz_computed = true;
  }
};


template <typename T>
class spectral : public derivative_3D<T,spectral<T>>{

public:
  enum class Direction{ x , y , z};
private:
  std::vector<T> kx,ky,kz;
  Direction direction;
  domain_size L;
public:
  using derivative_3D<T,spectral<T>>::_h;
  using derivative_3D<T,spectral<T>>::_G;
  using derivative_3D<T,spectral<T>>::_dx;
  using derivative_3D<T,spectral<T>>::_dy;
  using derivative_3D<T,spectral<T>>::_dz;
  using derivative_3D<T,spectral<T>>::_delta;
  using derivative_3D<T,spectral<T>>::_dx_computed;
  using derivative_3D<T,spectral<T>>::_dy_computed;
  using derivative_3D<T,spectral<T>>::_dz_computed;
  
  spectral(const step_size& h , const Grid3D& g,const domain_size D ,Direction dir) :
    derivative_3D<T,spectral<T>>(h, g)
    , direction(dir)
    ,L(D) {
    fill_wavenumber_x();
    fill_wavenumber_y();
    fill_wavenumber_z();
  }

  void fill_wavenumber_x() {
    if (direction == Direction::x){
      kx.resize((_G.NI)/2 + 1);
      for (std::size_t i = 0; i <= (_G.NI)/2; ++i)
	kx[i] = 2.0*M_PI*i / L.Lx;}
    else{
      for (std::size_t i = 0; i <= (_G.NI)/2; ++i){
	kx[i] = 2.0*M_PI*i / L.Lx;}
      for (std::size_t i = (_G.NI)/2 +1; i<_G.NI ; ++i){
	kx[i] = -2.0 * M_PI * (_G.NI - i) / L.Lx;}
    }
  }

  void fill_wavenumber_y() {
    if (direction == Direction::y){
      ky.resize((_G.NJ)/2 + 1);
      for (std::size_t j = 0; j <= (_G.NJ)/2; ++j)
        ky[j] = 2.0*M_PI*j / L.Ly;}
    else{
      for (std::size_t j = 0; j <= (_G.NJ)/2; ++j){
        ky[j] = 2.0*M_PI*j / L.Ly;}
      for (std::size_t j = (_G.NJ)/2 +1; j<_G.NJ ; ++j){
        ky[j] = -2.0 * M_PI * ( _G.NJ - j) / L.Ly;}
    }
  }

  void fill_wavenumber_z() {
    if (direction == Direction::z){
      kz.resize((_G.NK)/2 + 1);
      for (std::size_t k = 0; k <= (_G.NK)/2; ++k)
        kz[k] = 2.0*M_PI*k / L.Lz;}
    else{
      for (std::size_t k = 0; k <= (_G.NK)/2; ++k){
        kz[k] = 2.0*M_PI*k / L.Lz;}
      for (std::size_t k = (_G.NK)/2 +1; k<_G.NK ; ++k){
        kz[k] = -2.0 * M_PI * (_G.NK - k) / L.Lz;}
    }
  }

  template <typename Func>
  void second_der_x_impl(Func f) {
    double *in;
    fftw_complex *out;
    in  = (double*) fftw_malloc(sizeof(double) * (_G.NI * _G.NJ * _G.NK));
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((_G.NI * _G.NJ * _G.NK)
							      /2 + 1));
    for (std::size_t i=0; i<kx.size() ; ++i){
      for (std::size_t j=0; j<ky.size(); ++j){
	for (std::size_t k=0; k<kz.size(); ++k){
	  in[i*(_G.NJ * _G.NK) + j*(_G.NK) + k ] = f(i,j,k);
	  
	}
      }
    }
    
    fftw_plan plan_f = fftw_plan_dft_r2c_1d((_G.NI*_G.NJ*_G.NK), in, out, FFTW_ESTIMATE);
    fftw_execute(plan_f);
    for (std::size_t i = 0; i <= _G.NI/2; ++i)
      for (std::size_t j = 0; j < _G.NJ; ++j)
	for (std::size_t k = 0; k < _G.NK; ++k)
	  {
	    std::size_t idx = i*_G.NJ*_G.NK + j*_G.NK + k;
	    out[idx][0] = -kx[i]*kx[i]*out[idx][0];
	    out[idx][1] = -kx[i]*kx[i]*out[idx][1];  
	  }
    fftw_plan plan_b = fftw_plan_dft_c2r_3d(_G.NI, _G.NJ, _G.NK, out, in, FFTW_ESTIMATE);
    fftw_execute(plan_b);

    // 6. Store result in _dx
    for (std::size_t i = 0; i < _G.NI; ++i)
        for (std::size_t j = 0; j < _G.NJ; ++j)
            for (std::size_t k = 0; k < _G.NK; ++k)
                _dx(i,j,k) = in[i*_G.NJ*_G.NK + j*_G.NK + k] / (_G.NI*_G.NJ*_G.NK);

    // 7. Cleanup
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
    fftw_free(in);
    fftw_free(out);

    _dx_computed = true;
  }
  
  template <typename Func>
  void second_der_y_impl(Func f) {
    double *in;
    fftw_complex *out;
    in  = (double*) fftw_malloc(sizeof(double) * (_G.NI * _G.NJ * _G.NK));
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((_G.NI * _G.NJ * _G.NK)
							      /2 + 1));
    for (std::size_t i=0; i<kx.size() ; ++i){
      for (std::size_t j=0; j<ky.size(); ++j){
	for (std::size_t k=0; k<kz.size(); ++k){
	  in[i*(_G.NJ * _G.NK) + j*(_G.NK) + k ] = f(i,j,k);
	  
	}
      }
    }
    
    fftw_plan plan_f = fftw_plan_dft_r2c_1d((_G.NI*_G.NJ*_G.NK), in, out, FFTW_ESTIMATE);
    fftw_execute(plan_f);
    for (std::size_t i = 0; i <= _G.NI/2; ++i)
      for (std::size_t j = 0; j < _G.NJ; ++j)
	for (std::size_t k = 0; k < _G.NK; ++k)
	  {
	    std::size_t idx = i*_G.NJ*_G.NK + j*_G.NK + k;
	    out[idx][0] = -ky[i]*ky[i]*out[idx][0];
	    out[idx][1] = -ky[i]*ky[i]*out[idx][1];  
	  }
    fftw_plan plan_b = fftw_plan_dft_c2r_3d(_G.NI, _G.NJ, _G.NK, out, in, FFTW_ESTIMATE);
    fftw_execute(plan_b);

    // 6. Store result in _dx
    for (std::size_t i = 0; i < _G.NI; ++i)
        for (std::size_t j = 0; j < _G.NJ; ++j)
            for (std::size_t k = 0; k < _G.NK; ++k)
                _dy(i,j,k) = in[i*_G.NJ*_G.NK + j*_G.NK + k] / (_G.NI*_G.NJ*_G.NK);

    // 7. Cleanup
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
    fftw_free(in);
    fftw_free(out);

    _dy_computed = true;
  }
  
  template <typename Func>
  void second_der_z_impl(Func f) {
    double *in;
    fftw_complex *out;
    in  = (double*) fftw_malloc(sizeof(double) * (_G.NI * _G.NJ * _G.NK));
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((_G.NI * _G.NJ * _G.NK)
							      /2 + 1));
    for (std::size_t i=0; i<kx.size() ; ++i){
      for (std::size_t j=0; j<ky.size(); ++j){
	for (std::size_t k=0; k<kz.size(); ++k){
	  in[i*(_G.NJ * _G.NK) + j*(_G.NK) + k ] = f(i,j,k);
	  
	}
      }
    }
    
    fftw_plan plan_f = fftw_plan_dft_r2c_1d((_G.NI*_G.NJ*_G.NK), in, out, FFTW_ESTIMATE);
    fftw_execute(plan_f);
    for (std::size_t i = 0; i <= _G.NI/2; ++i)
      for (std::size_t j = 0; j < _G.NJ; ++j)
	for (std::size_t k = 0; k < _G.NK; ++k)
	  {
	    std::size_t idx = i*_G.NJ*_G.NK + j*_G.NK + k;
	    out[idx][0] = -kz[i]*kz[i]*out[idx][0];
	    out[idx][1] = -kz[i]*kz[i]*out[idx][1];  
	  }
    fftw_plan plan_b = fftw_plan_dft_c2r_3d(_G.NI, _G.NJ, _G.NK, out, in, FFTW_ESTIMATE);
    fftw_execute(plan_b);

    // 6. Store result in _dx
    for (std::size_t i = 0; i < _G.NI; ++i)
        for (std::size_t j = 0; j < _G.NJ; ++j)
            for (std::size_t k = 0; k < _G.NK; ++k)
                _dz(i,j,k) = in[i*_G.NJ*_G.NK + j*_G.NK + k] / (_G.NI*_G.NJ*_G.NK);

    // 7. Cleanup
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
    fftw_free(in);
    fftw_free(out);

    _dz_computed = true;
  }

};
