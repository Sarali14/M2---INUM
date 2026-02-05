#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <mpi.h>

using namespace Eigen;
int main(int argc , char *argv[]){
  int nb_procs, rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD , &nb_procs);
  MPI_Comm_rank(MPI_COMM_WORLD , &rank);

  int N = 32 ;
  double L = 1.0;
  double h = L/(N+1);
  double c = 0;
  double invh2 = 1.0/(h*h);
  double u_a = 0;
  double u_b = 0;
  //nb de points pour le recouvrement
  int rc = 1 ;
  
  int N_loc = N / nb_procs;

  int other = 1 - rank;

  MatrixXd A(N_loc+1+rc,N_loc+1+rc);
  A.setZero();
  VectorXd f(N_loc+1+rc);
  f.setZero();  

  for (std::size_t i=0; i < N_loc+1+rc; i++){
    A(i,i) = 2.0 * invh2 + c;
    f(i) = 2.0;
    if (i > 0)
      A(i, i - 1) = -invh2;

    if (i < N_loc + 1)
      A(i, i + 1) = -invh2;
  }
  A.row(0).setZero();
  A(0,0) = 1.0;
  A.row(N_loc+rc).setZero();
  A(N_loc+rc,N_loc+rc) = 1.0;

  
  double u_right = 0.0;
  double u_left = 0.0;
  
  if(rank == 0){
    f(0) = u_a ;
    f(N_loc+rc) = u_right;
  }

  if(rank == 1){
    f(N_loc+rc) = u_b ;
    f(0) = u_left;
  }

  int kmax = 1000;
  double tol = 1e-6;
  int k = 0;
  
  MatrixXd Ainv = A.inverse();
  VectorXd u_k(N_loc+1+rc);
  
  u_k.setZero();  
  u_k = Ainv * f;
  // double err = 0.0;

  double send_left = u_k(0);
  double send_right = u_k(N_loc+rc);
 
  MPI_Sendrecv(&send_left, 1, MPI_DOUBLE , 1, 10,
	       &u_right, 1, MPI_DOUBLE, 1, 10,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&send_right, 1, MPI_DOUBLE , 0, 10,
	       &u_left, 1, MPI_DOUBLE, 0, 10,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  double pt_left = u_k(N_loc+rc-1);
  double pt_right = u_k(1);
  
  double local_err = 0.0;
  if ( rank == 0 ){
    local_err = - u_k(N_loc+rc) + pt_left;
  }
  if ( rank == 1 ){
    local_err = - u_k(0)+ pt_right;
  }

  double err = 0.0;

  MPI_Allreduce(&local_err , &err, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    
  while (std::abs(err) > tol &&  k < kmax){
    std::cout << "HI";
    double send_left = u_k(0);
    double send_right = u_k(N_loc+rc);

    double pt_left = u_k(N_loc+rc-1);
    double pt_right = u_k(1);

    MPI_Sendrecv(&send_left, 1, MPI_DOUBLE , 1, 10,
		 &u_right, 1, MPI_DOUBLE, 1, 10,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&send_right, 1, MPI_DOUBLE , 0, 10,
		 &u_left, 1, MPI_DOUBLE, 0, 10,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  if ( rank == 0 ){
    local_err = - u_k(N_loc+rc) + pt_left;
  }
  if ( rank == 1 ){
    local_err = - u_k(0)+ pt_right;
  }

  double err = 0.0;

  MPI_Allreduce(&local_err , &err, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  
    if(rank == 0){
      f(0) = u_a ;
      f(N_loc+rc) = u_right;
    }

    if(rank == 1){
      f(N_loc+rc) = u_b ;
      f(0) = u_left;
    }
    u_k = Ainv * f;    
    k = k+1;
  }
  std::cout <<"iteration : "<<k<<std::endl;
  
  std::cout <<"erreur = "<< err <<std::endl;
  /*{
    std::ofstream out;
    out.open("solution_rank"+std::to_string(rank)+".dat");

    for (std::size_t i = 0; i < N_loc+rc+1; ++i) {                                       
      double x = (double)(rank*N_loc + i) * h;                                   
      out << x << " " << u_k((int)i) << "\n";
    }
    out.close();
    }*/
  MPI_Finalize();
  
  return 0;
}
