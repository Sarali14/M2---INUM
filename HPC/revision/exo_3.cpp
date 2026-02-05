#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <Eigen/Dense>
#include <fstream>

using namespace Eigen;
int main ( int argc , char *argv[]){
  int nb_procs , rank;
  double L = M_PI;
  MPI_Init(&argc , &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int N = std::atoi(argv[1]);
  int a = std::atoi(argv[2]);
  int k_max = std::atoi(argv[3]);
  double tol = std::atof(argv[4]);
  
  int me = rank;
  int neighbor_right = (me + 1)% nb_procs;
  int neighbor_left = (me -1 +nb_procs)%nb_procs;

  int n = N / nb_procs;
  double h = L/(N-1);
  double invh2 = 1/(h*h);
  
  MatrixXd A;
  VectorXd b;
  A.resize(n,n);
  b.resize(n);
  A.setZero();
  
  for (int i= 0 ; i< n ; i++){
    A(i,i) = 2*invh2+a;
    int ig = rank * n + i;
    b(i) = std::sin(2.0 * M_PI * (ig*h));
    if ( i > 0)
      A(i,i-1) = -invh2;
    if ( i < n-1)
      A(i,i+1) = -invh2;
  }

  VectorXd u(n);
  VectorXd r(n);
  double rs,rs_new,alpha,beta;
  VectorXd pk(n);
  VectorXd A_pk(n);
  u.setZero();
  r = b - A * u;
  pk = r ;
  double rs_loc = r.dot(r);
  MPI_Allreduce(&rs_loc, &rs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  int k = 0;
  while (sqrt(rs)>tol && k < k_max){
    double pk_left_ghost = 0.0;
    double pk_right_ghost = 0.0;

    double send_left  = pk(0);
    double send_right = pk(n-1);

    MPI_Sendrecv(&send_left,  1, MPI_DOUBLE, neighbor_left,  0,
		 &pk_right_ghost, 1, MPI_DOUBLE, neighbor_right, 0,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, neighbor_right, 1,
		 &pk_left_ghost,  1, MPI_DOUBLE, neighbor_left,  1,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    A_pk = A * pk;
    A_pk(0)     += -invh2 * pk_left_ghost;
    A_pk(n - 1) += -invh2 * pk_right_ghost;

    double pAp_local = pk.dot(A_pk);
    double pAp = 0.0;
    MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    alpha = rs / pAp;
    
    u = u + alpha * pk;
    r = r- alpha * A_pk;
    double rs_new_loc = r.dot(r);
    MPI_Allreduce(&rs_new_loc , & rs_new, 1 , MPI_DOUBLE, MPI_SUM ,MPI_COMM_WORLD);
    beta = rs_new /rs;
    pk = r + beta * pk;
    rs = rs_new;
    k +=1;
  }
  if (rank == 0) {
    std::cout << "CG converged in " << k << " iterations\n";
    std::cout << "Final residual norm = " << std::sqrt(rs) << std::endl;
  }
  {
    std::ofstream out;
    out.open("solution_rank" + std::to_string(rank) + ".dat");

    for (std::size_t i = 0; i < n; ++i) {
      std::size_t j = rank*n + i;     // global index                                       
      double x = (double)j * h;      // global coordinate                                  
      out << x << " " << u((int)i) << "\n";
    }

    out.close();
  }

  MPI_Finalize();
  return 0;
}
