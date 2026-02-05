#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <iomanip>

int main (int argc, char *argv[]){
  int nb_proc,rank;
  
  MPI_Init(&argc , &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD,&nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int N = 4;
  int M = 4;
  std::vector<double> A(N*M , 0.0);

  if (rank < M) {
    int i = rank; // global row index owned by this rank

    // 3) Fill only that row (iteratively)
    // Example: put 0..15 row-major if M=N=4
    for (int j = 0; j < N; ++j) {
      A[i* N + j] = i * N + j;
    }
  }

  // Optional: show what each rank has (they differ)
  for (int r = 0; r < nb_proc; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
      std::cout << "\nRank " << rank << " matrix:\n";
      for (int i = 0; i < M; ++i) {
	for (int j = 0; j < N; ++j) {
	  std::cout << std::setw(5) << A[i * N + j] << " ";
	}
	std::cout << "\n";
      }
      std::cout << std::flush;
    }
  }

  // Temporary buffer to receive a row
    std::vector<double> recv_row(N);

    // 3) ring: at each step, send one row and receive one row
    // Initially, the "current row to send" is my own row.
    int send_row_index = rank;

    int left  = (rank - 1 + nb_proc) % nb_proc;
    int right = (rank + 1) % nb_proc;

    // The "row currently being forwarded" (initially my own row)
    int send_row = rank;

    for (int step = 0; step < nb_proc - 1; ++step) {

      // Row I will receive this step (comes from left, shifts each time)
      int recv_row = (rank - 1 - step + nb_proc) % nb_proc;

      MPI_Sendrecv(
		   /*sendbuf*/ &A[send_row * N], N, MPI_DOUBLE, right, 100,
		   /*recvbuf*/ &A[recv_row * N], N, MPI_DOUBLE, left,  100,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE
		   );

      // Next step, forward the row I just received
      send_row = recv_row;
    }



    // Now every rank has all rows => matrices identical on all ranks.

    for (int r = 0; r < nb_proc; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == r) {
	std::cout << "\nFinal matrix on rank " << rank << ":\n";
	for (int i = 0; i < M; ++i) {
	  for (int j = 0; j < N; ++j) {
	    std::cout << std::setw(5) << A[i * N + j] << " ";
	  }
	  std::cout << "\n";
	}
	std::cout << std::flush;
      }
    }

  MPI_Finalize();
  return 0;
}
