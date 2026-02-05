#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>

using namespace Eigen;
int main(){
  
  int N = 32 ;
  double L = 1.0;
  double h = L/(N+1);

  MatrixXd A(N,N);
  A.setZero();
  VectorXd f(N);
  f.setZero();
  double c = 1;
  double invh2 = 1.0/(h*h);

  for (std::size_t i=0; i < N; i++){
    A(i,i) = 2.0 * invh2 + c;
    f(i) = 2.0; // TERME SOURCE POUR LA SOLUTION PARABOLIQUE
    if (i > 0)
      A(i, i - 1) = -invh2;

    if (i < N - 1)
      A(i, i + 1) = -invh2;
  }

  //f(N-1) += invh2; CONDITION AU BORD DROITE POUR LE TERME SOURCE 0

  VectorXd u(N);
  u = A.inverse() * f;

  {
    std::ofstream out;
    out.open("solution_rank.dat");

    for (std::size_t i = 0; i < N; ++i) {                                       
      double x = (double)i * h;                                   
      out << x << " " << u((int)i) << "\n";
    }
    out.close();
  }
  return 0;
}

  
