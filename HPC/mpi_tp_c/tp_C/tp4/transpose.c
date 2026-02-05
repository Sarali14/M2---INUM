#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int nb_lignes=4;
  int nb_colonnes=5;
  int nb_lignes_t=nb_colonnes;
  int nb_colonnes_t=nb_lignes;
  int etiquette=1000;
  double a[nb_lignes][nb_colonnes];
  double at[nb_lignes_t][nb_colonnes_t];
  int rang,iterl,iterc,taille_reel;
  MPI_Datatype type_colonne, type_transpose;
  MPI_Aint pas;
  MPI_Status statut;

  /* TODO: Recuperation du rang -> rang */

  /* TODO: Type_transpose */


  if (rang == 0) {
    /* Initialisation de A */
    for (iterl=0; iterl<nb_lignes; iterl++)
      for (iterc=0; iterc<nb_colonnes; iterc++)
        a[iterl][iterc] = 1+iterl*nb_colonnes+iterc;
    /* Affichage de A */
    printf("Matrice a\n");
    for (iterl=0; iterl<nb_lignes;iterl++) {
      for (iterc=0; iterc<nb_colonnes; iterc++) {
        printf("%4.f ", a[iterl][iterc]);
      }
      printf("\n");
    }
    /* TODO: Envoi de la matrice A au processus 1 */

  } else {
    /* TODO: Reception dans la matrice AT */

    /* Affichage */
    printf("Matrice transposee at\n");
    for (iterc=0; iterc<nb_lignes_t; iterc++) {
      for (iterl=0; iterl<nb_colonnes_t;iterl++) {
        printf("%4.f ", at[iterc][iterl]);
      }
      printf("\n");
    }
  }

  /* TODO : Nettoyage types MPI */

  /* TODO: Sortie de MPI */

  return 0;
}
