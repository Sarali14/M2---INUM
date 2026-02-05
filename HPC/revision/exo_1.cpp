#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <random>
#include <ctime>

int main(int argc , char *argv[]){
  int nb_proc , rank;
  int N=1000 , etiquette_ping = 01, etiquette_pong= 10;
  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD , &nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD , &rank);
  
  std::vector<int> v(N);

  srand(time(0));
  for (std::size_t i =0 ; i < N ; i++){
    int a = rand() % 1001;
    v[i] = a;
  }

  double start = MPI_Wtime();
  if( rank == 0){
    MPI_Send(v.data(), 1000, MPI_INT, 1, etiquette_ping , MPI_COMM_WORLD);
  }
  else if (rank == 1){
    MPI_Recv(v.data(), 1000, MPI_INT, 0, etiquette_ping , MPI_COMM_WORLD , &status);
  }
  std::cout << "PING"<<std::endl;

  if( rank == 1){
    MPI_Send(v.data(), 1000, MPI_INT, 0, etiquette_pong , MPI_COMM_WORLD);
  }
  else if (rank == 0){
    MPI_Recv(v.data(), 1000, MPI_INT, 1, etiquette_pong , MPI_COMM_WORLD , &status);
  }
  std::cout << "PONG"<<std::endl;
  double end = MPI_Wtime();

  std::cout << "elapsed time = "<<end-start<<std::endl;
  MPI_Finalize();

  return 0;
}

  

  

  
    
  
