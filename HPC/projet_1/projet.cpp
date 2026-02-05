#include <iostream>
#include <mpi.h>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
using namespace Eigen;

int main(int argc,char *argv[]) {
  MPI_Init(&argc, &argv);
  int size,rank;
 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cerr << "Hello from rank=" << rank << " size=" << size << "\n";

  if (size != 2) {
    if (rank == 0) std::cerr << "Run with: mpirun -np 2 ./a.out\n";
    MPI_Finalize();
    return 1;
  }

  std::size_t N=32;
  double L = 2*M_PI;
  int maxIters = 20000;
  double tol = 1e-6;
  double h = L / (double)(size*N);

  double omega = 0.9 / (1.0 + 4.0/(h*h));
  const double invh2 = 1.0 / (h * h);
  
  std::size_t start = N * (std::size_t)rank;

  VectorXd u = VectorXd::Zero(N);
  VectorXd f = VectorXd::Zero(N);
  MatrixXd A = MatrixXd::Zero(N,N);
  VectorXd b = VectorXd::Zero(N);
  VectorXd r = VectorXd::Zero(N);

  for (std::size_t i =0; i<N; ++i){

    A(i,i) = -2.0*invh2 + 1;
    if (i>0)
      A(i,i-1) = invh2;
    if (i<N-1)
      A(i,i+1) = invh2;
    
    std::size_t j = start + i;
    f(i)= -3.0*std::sin(2.0*(j*h));
  }

  int other = 1 - rank;
 
  double f2_local = f.squaredNorm();
  double f2_other = 0.0;

  MPI_Sendrecv(&f2_local, 1, MPI_DOUBLE, other, 40,
	       &f2_other, 1, MPI_DOUBLE, other, 40,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  double fnorm_global = std::sqrt(std::max(f2_local + f2_other, 1e-30));

  for (int k = 0; k < maxIters; ++k) {

    double uLghost = 0.0, uRghost = 0.0;

    double send_left  = u(0);         
    double send_right = u(N - 1); 

    MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, other, 10,
                 &uRghost,   1, MPI_DOUBLE, other, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, other, 20,
                 &uLghost,    1, MPI_DOUBLE, other, 20,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    b = A * u;
    b(0)   += invh2*uLghost;
    b(N-1) += invh2*uRghost;

    r = f - b;

    double r2_local = r.squaredNorm();
    double r2_other = 0.0;

    MPI_Sendrecv(&r2_local, 1, MPI_DOUBLE, other, 30,
                 &r2_other, 1, MPI_DOUBLE, other, 30,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double rnorm_global = std::sqrt(r2_local + r2_other);

    double relres = rnorm_global / fnorm_global;

    if (rank == 0 && (k % 2000 == 0)) {
      std::cout << "iter " << k << " relres " << relres << "\n";
    }

    if (relres <= tol) {
      if (rank == 0) {
	std::cout << "Converged at iter " << k
		  << " with relres " << relres << "\n";
      }
      break;
    }

    u.noalias() += omega * r;
  }
  {
    std::ofstream out;
    out.open("solution_rank" + std::to_string(rank) + ".dat");

    for (std::size_t i = 0; i < N; ++i) {
      std::size_t j = start + i;     // global index
      double x = (double)j * h;      // global coordinate
      out << x << " " << u((int)i) << "\n";
    }

    out.close();
  }
  
  MPI_Finalize();
  return 0;
} 
