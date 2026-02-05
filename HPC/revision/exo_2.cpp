#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <string>

double f(double x){
  return 4/(1+x*x);
}
int main (int argc , char *argv[]){
  int nb_procs, rank;

  MPI_Init(&argc , &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (argc != 5) {
    if(rank== 0){  std::cerr << "Usage: " << argv[0]
	      << " <method: rectR | rectL | rectM | Trap>\n"
	      << " <number of discretization points : N> \n"
	      << " <boundaries of the interval [a,b]> \n" ;
    return 1;
    }
  }

  std::string method = argv[1];
  int N = std::atoi(argv[2]);
  int a = std::atoi(argv[3]);
  int b = std::atoi(argv[4]);

  double h = (double)(b - a)/(double)N;
  int size_subdomain = N/nb_procs;  
  std::size_t start = rank*size_subdomain;
  std::size_t end = start +size_subdomain;
  
  if (method == "rectR") {

    double local_sum = 0;
    for (int i = start ; i < end ; i++){
      double x  = h * ( i + 1 );
      local_sum = local_sum + h * f(x);
    }

    double global_sum = 0;
    MPI_Allreduce( &local_sum , &global_sum , 1 , MPI_DOUBLE ,MPI_SUM, MPI_COMM_WORLD);

    if ( rank == 0 ) {
      std::cout <<method<<std::endl <<"le résultat de l'approximation  est  : "<< global_sum <<std::endl;
      std::cout << "erreur absolue pour " <<N<< " points est : " <<std::abs(M_PI - global_sum)<<std::endl;
    }
  }
  else if (method == "rectL") {
    double local_sum = 0;
    for (int i = start ; i < end ; i++){
      double x  = h * ( i );
      local_sum = local_sum + h * f(x);
    }

    double global_sum = 0;
    MPI_Allreduce( &local_sum , &global_sum , 1 , MPI_DOUBLE ,MPI_SUM, MPI_COMM_WORLD);

    if ( rank == 0 ) {
      std::cout <<method<<std::endl <<"le résultat de l'approximation est  : "<< global_sum <<std::endl;
      std::cout << "erreur absolue pour " <<N<< " points est : " <<std::abs(M_PI - global_sum)<<std::endl;
    }
  }
  else if (method == "rectM") {
    double local_sum = 0;
    for (int i = start ; i < end ; i++){
      double x  = h * ( i + 0.5 );
      local_sum = local_sum + h * f(x);
    }

    double global_sum = 0;
    MPI_Allreduce( &local_sum , &global_sum , 1 , MPI_DOUBLE ,MPI_SUM, MPI_COMM_WORLD);

    if ( rank == 0 ) {
      std::cout <<method<<std::endl<<"le résultat de l'approximation est  : "<< global_sum <<std::endl;
      std::cout << "erreur absolue pour " <<N<< " points est : " <<std::abs(M_PI - global_sum)<<std::endl;
    }
  }
  else if (method == "Trap") {
    double local_sum = 0;
    if ( rank == 0 ) start +=1;
    else if (rank == nb_procs) end += -1;
    
    for (int i = start ; i < end ; i++){
      double x  = a + h * i;
      local_sum = local_sum + h * f(x);
    }

    double global_sum = 0;
    MPI_Allreduce( &local_sum , &global_sum , 1 , MPI_DOUBLE ,MPI_SUM, MPI_COMM_WORLD);

    global_sum = global_sum + 0.5*h*(f(a)+f(b));
    if ( rank == 0 ) {
      std::cout <<method<<std::endl<<"le résultat de l'approximation est  : "<< global_sum <<std::endl;
      std::cout << "erreur absolue pour " <<N<< " points est : " <<std::abs(M_PI - global_sum)<<std::endl;
    } 
  }
  else {
    std::cerr << "Unknown method: " << method << "\n";
    return 1;
  }
  
  
  MPI_Finalize();
  return 0;
}
