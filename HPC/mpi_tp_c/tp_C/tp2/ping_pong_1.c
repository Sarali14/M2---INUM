#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int rang,iter;
  int nb_valeurs=1000;
  int etiquette=99;
  double valeurs[nb_valeurs];
  MPI_Status statut;

  /* TODO: Initialisation MPI & recuperation du rang */

  if (rang == 0) {
    /* Remplir avec des nombres aleatoires */
    for (iter = 0; iter<nb_valeurs; iter++)
      valeurs[iter] = rand() / (RAND_MAX + 1.);
    /* TODO : Processus 0 envoie valeurs au processus 1 */

  } else if(rang == 1) {
    /* TODO: Processus 1 recoit valeurs du processus 0 */

    printf("Moi, processus 1, j'ai recu %d valeurs (derniere = %g)"
           "du processus 0.\n", nb_valeurs, valeurs[nb_valeurs-1]);
  }

  /* TODO: Finaliser MPI */
  
  return 0;
}
