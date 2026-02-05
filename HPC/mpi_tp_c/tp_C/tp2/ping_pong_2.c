#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int rang,iter;
  int nb_valeurs=1000;
  int etiquette=99;
  double valeurs[nb_valeurs];
  MPI_Status statut;
  double temps_debut,temps_fin;

  /* TODO Recuperation du rang -> rang */

  if (rang == 0) {
    for (iter = 0; iter<nb_valeurs; iter++)
      valeurs[iter] = rand() / (RAND_MAX + 1.);
    /* TODO: Debut de prise de temps du ping pong -> temps_debut */

    /* TODO : Processus 0: envoie et recoit du processus 1 */

    /* TODO : Fin de prise de temps du ping pong -> temps_fin */

    printf("Moi, processus 0, j'ai envoye et recu %d valeurs"
           "(derniere = %g) du processus 1 en %f secondes.\n",
           nb_valeurs, valeurs[nb_valeurs-1], temps_fin-temps_debut);
  } else {
    /* TODO: Processus 1: recoit et envoie au processus 0 */

  }

  /* TODO: Finaliser MPI */
  
  return 0;
}
