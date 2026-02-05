#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char * argv[]) {
  const int nb_valeurs=121;
  int valeurs[nb_valeurs];
  int rang,iter;
  MPI_File descripteur;
  int nb_octets_entier;
  MPI_Offset position_fichier;
  MPI_Status statut;
  char nom_fichier[256];
  FILE * fichier;

  MPI_Init( &argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rang);
  MPI_File_set_errhandler(MPI_FILE_NULL,MPI_ERRORS_ARE_FATAL);

  /* TODO: Ouverture du fichier donnees.dat en lecture */

  for (iter=0;iter<nb_valeurs; iter++) valeurs[iter]=0;
  /* TODO: Lecture via des deplacements explictites en mode individuel */

  sprintf(nom_fichier,"fichier_dei%1d.dat",rang);
  fichier = fopen(nom_fichier,"w");
  for (iter=0; iter<nb_valeurs; iter++)
    fprintf(fichier,"%3d\n",valeurs[iter]);
  fclose(fichier);

  for (iter=0;iter<nb_valeurs; iter++) valeurs[iter]=0;
  /* TODO: Lecture via les pointeurs partages en mode collectif */

  sprintf(nom_fichier,"fichier_ppc%1d.dat",rang);
  fichier = fopen(nom_fichier,"w");
  for (iter=0; iter<nb_valeurs; iter++)
    fprintf(fichier,"%3d\n",valeurs[iter]);
  fclose(fichier);

  for (iter=0;iter<nb_valeurs; iter++) valeurs[iter]=0;
  /* TODO: Lecture via les pointeurs individuels en mode individuel */

  sprintf(nom_fichier,"fichier_pii%1d.dat",rang);
  fichier = fopen(nom_fichier,"w");
  for (iter=0; iter<nb_valeurs; iter++)
    fprintf(fichier,"%3d\n",valeurs[iter]);
  fclose(fichier);

  for (iter=0;iter<nb_valeurs; iter++) valeurs[iter]=0;
  /* TODO: Lecture via les pointeurs partages en mode individuel */

  sprintf(nom_fichier,"fichier_ppi%1d.dat",rang);
  fichier = fopen(nom_fichier,"w");
  for (iter=0; iter<nb_valeurs; iter++)
    fprintf(fichier,"%3d\n",valeurs[iter]);
  fclose(fichier);

  /* TODO: Fermeture du fichier */

  MPI_Finalize();
  return 0;
}
