#include <iostream>
#include <cmath>
#include <random>
#include <mpi.h>
#include <vector>

int main(int argc , char *argv[]){
  int nb_procs,rank;
  int values[9] = {1,6,3,8,5,6,8,0,7};
  int etiquette_ping = 01;
  int etiquette_pong = 10;
  int nb_turns = sizeof(values)/sizeof(values[0]);
  std::size_t N;
  std::vector<int> v;
  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nb_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  for ( int i = 0 ; i < nb_turns ; ++ i ){
    N = values[i];
    v.resize(N);
    for (std::size_t j = 0 ; j<N ; j++){
      int a = rand()%1001;
      v[j] = a;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if( rank == 0){
      MPI_Send(v.data(), v.size(), MPI_INT, 1, etiquette_ping , MPI_COMM_WORLD);
    }
    else if (rank == 1){
      MPI_Recv(v.data(), v.size(), MPI_INT, 0, etiquette_ping , MPI_COMM_WORLD , &status);
    }
    if(rank==0){
      std::cout << "PING"<<std::endl;}

    if( rank == 1){
      MPI_Send(v.data(), v.size(), MPI_INT, 0, etiquette_pong , MPI_COMM_WORLD);
    }
    else if (rank == 0){
      MPI_Recv(v.data(), v.size(), MPI_INT, 1, etiquette_pong , MPI_COMM_WORLD , &status);
    }
    if(rank==0){
      std::cout << "PONG"<<std::endl;}
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if (rank == 0){
    std::cout << "elapsed time = "<<end-start<<std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}

  
  
