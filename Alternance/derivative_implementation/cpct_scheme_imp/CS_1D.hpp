#pragma once

#include<iostream>
#include<cmath>
#include<vector>
#include "vector_2D.hpp"

template <typename T>
class CS1D{
protected:
  double _h; //step size
  std::size_t _N; //nb of discretization points
  std::vector<T> _d; //solution vector
  Grid2D _size; 
  Array2D<T> _A; //LHS coefficient matrix
  Array2D<T> _B; //RHS coefficient matrix
  std::vector<T> _rhs; //computed RHS with function vector
  double _alpha; //coefficient for (i+1)th and (i-1)th derivatives
  double _beta; //coefficient for (i+2)th and (i-2)th derivatives, will always be zero as we will be restricted to tri-diag systems 
  double _a; //coefficient for central FD scheme for (i-1) & (i+1) rhs
  double _b; //coefficient for central FD scheme for (i-2) & (i+2) rhs
  double _c; //coefficient for central FD scheme for (i-3) & (i+3) rhs

public:
  CS1D(double h, std::size_t N):
    _h(h), _N(N), _size{N,N}, _d(N), _A(_size),_B(_size),_rhs(N), _beta(0) {}

  ~CS1D() {}

  const double& get_step_size() const{return _h;}
  const std::size_t& get_nb_nodes() const{return _N;}
  const std::vector<T>& get_derivative() {return _d;}
  const double& get_alpha() {return _alpha;}
  const double& get_beta() {return _beta;}
  const double& get_a() {return _a;}
  const double& get_b() {return _b;}
  const double& get_c() {return _c;}


  void init_A_first_der() {
    _A.init_zero();  
    for (std::size_t i=0;i<_N;++i){
       _A(i,i)=1;
       _A(i,(i-1+_N)%_N)=_alpha;
       _A(i,(i+1+_N)%_N)=_alpha;
       _A(i,(i-2+_N)%_N)=_beta;
       _A(i,(i+2+_N)%_N)=_beta;
    }
  }

  void init_B_first_der() {
    _B.init_zero();
    for (std::size_t i=0;i<_N;++i){
       _B(i,(i-1+_N)%_N)=-6*_a;
       _B(i,(i+1)%_N)=6*_a;
       _B(i,(i-2+_N)%_N)=-3*_b;
       _B(i,(i+2)%_N)=3*_b;
       _B(i,(i-3+_N)%_N)=-2*_c;
       _B(i,(i+3)%_N)=2*_c;
    }
  }
  template <typename Func>
  void compute_rhs_first_der(Func f){
    for (std::size_t i=0;i<_N;++i){
       _rhs[i]=_B(i,(i-3+_N)%_N)*f(_h*((i-3+_N)%_N))+
               _B(i,(i-2+_N)%_N)*f(_h*((i-2+_N)%_N))+
               _B(i,(i-1+_N)%_N)*f(_h*((i-1+_N)%_N))+
               _B(i,i)*f(_h*(i))+
               _B(i,(i+1)%_N)*f(_h*((i+1)%_N))+
               _B(i,(i+2)%_N)*f(_h*((i+2)%_N))+
               _B(i,(i+3)%_N)*f(_h*((i+3)%_N));
    }
  }

  void thomas_solver(){
    for (std::size_t i=0;i<2;++i){ //first loop to normalize the first 2 rows that will be used for computations
       _d[i]=_rhs[i]/_A(i,i); //store the normalized rhs TEMPORARILY in the solution vector, to avoid allocating a new vector
       _A(i,(i+1)%_N)=_A(i,(i+1)%_N)/_A(i,i); //upper diagonal normalized
       _A(i,(i-1+_N)%_N)=_A(i,(i-1+_N)%_N)/_A(i,i); //lower diagonal normalized
    }
    T w;
    for (std::size_t i=2;i<_N;++i){ //forward pass
      w=1/(_A(i,i)-_A(i-1,(i+1)%_N)*_A(i,(i-1+_N)%_N));
       _d[i]=w*(_rhs[i]-_A(i,(i-1+_N)%_N)*_d[i-1]);
       _A(i,(i-1+_N)%_N)=-w*_A(i,(i-1+_N)%_N)*_A(i-1,(i-1+_N)%_N);
       _A(i,(i+1)%_N)=w*_A(i,(i+1)%_N);
    }
    
    for (std::size_t i=_N-3;i>1;--i){ //backward pass
       _d[i]=_d[i]-_A(i,(i+1)%_N)*_d[i+1];
       _A(i,(i-1+_N)%_N)=_A(i,(i-1+_N)%_N)-_A(i,(i+1)%_N)*_A(i+1,(i-1+_N)%_N);
       _A(i,(i+1)%_N)=-_A(i,(i+1)%_N)*_A(i+1,(i+1)%_N);
    }
    w=1/(1-_A(0,1)*_A(1,0));
    _d[0]=w*(_d[0]-_A(0,1)*_d[1]);
    _A(1,0)=w*_A(0,1);
    _A(0,1)=-w*_A(0,1)*_A(1,2);
    
    for (std::size_t i=1;i<_N-2;++i){ //substitution
       _d[i]=_d[i]-_A(i,(i-1+_N)%_N)*_d[0]-_A(i,(i+1)%_N)*_d[_N-1];
     }
    }  
  template <typename Func>
  void run_algo(Func f){
    init_A_first_der();
    init_B_first_der();
    compute_rhs_first_der(f);
    thomas_solver();
  }
};

template <typename T>
class CS1D_o2 : public CS1D<T> {

public:
    
    CS1D_o2(double h, std::size_t N, double alpha=1.0/4.0) :  CS1D<T>(h,N) {
        this->_alpha=alpha;
        this->_a=1+2*(alpha);
        this->_b=0;
        this->_c=0;
    }
};

template <typename T>
class CS1D_o4 : public CS1D<T> {

public:
    
    CS1D_o4(double h, std::size_t N, double alpha=1.0/3.0) :  CS1D<T>(h,N) {
        this->_alpha=alpha;
        this->_a=(2/3)*(alpha+2);
        this->_b=(1/3)*(4*alpha -1);
        this->_c=0;
    }
};

template <typename T>
class CS1D_o6 : public CS1D<T> {

public:
    
    CS1D_o6(double h, std::size_t N, double alpha=1.0/3.0) :  CS1D<T>(h,N) {
        this->_alpha=alpha;
        this->_a=(1/6)*(alpha+9);
        this->_b=(1/15)*(32*alpha -9);
        this->_c=(1/10)*(-3*alpha +1);
    }
};

template <typename T>
class CS1D_o8 : public CS1D<T> {

public:
    
    CS1D_o8(double h, std::size_t N, double alpha=3/8) :  CS1D<T>(h,N) {
        this->_alpha=3/8;
        this->_a=(2/3)*(alpha+2);
        this->_b=(1/3)*(4*alpha -1);
        this->_c=0;
    }
};



