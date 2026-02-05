#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int rang,iter;
  int nb_valeurs=1000;
  int etiquette=99;
  double valeurs[nb_valeurs];
  MPI_Status statut;

  /* Initialisation MPI & recuperation du rang */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rang);

  if (rang == 0) {
    /* Remplir avec des nombres aleatoires */
    for (iter = 0; iter<nb_valeurs; iter++)
      valeurs[iter] = rand() / (RAND_MAX + 1.);
    /* Processus 0 envoie valeurs au processus 1 */
    MPI_Send(valeurs,nb_valeurs,MPI_DOUBLE,1,etiquette,MPI_COMM_WORLD);
  } else if(rang == 1) {
    /* Processus 1 recoit valeurs du processus 0 */
    MPI_Recv(valeurs,nb_valeurs,MPI_DOUBLE,0,etiquette,MPI_COMM_WORLD,&statut);
    printf("Moi, processus 1, j'ai recu %d valeurs (derniere = %g)"
           "du processus 0.\n", nb_valeurs, valeurs[nb_valeurs-1]);
  }

  /* Finaliser MPI */
  MPI_Finalize();
  return 0;
}
