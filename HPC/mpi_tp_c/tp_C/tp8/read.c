#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

int main(int argc, char *argv[]) {
  int rang;
  int ntx, nty;
  FILE *fichier;
  double *u;
  int iter;
  MPI_File descripteur;
  int code;
  MPI_Status statut;
  MPI_Offset taille_fichier;
  int taille_reel;
  int texte_longueur;
  char texte_erreur[MPI_MAX_ERROR_STRING];

  MPI_Init( &argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rang);
  fichier = fopen("poisson.data", "r");
  if (fscanf(fichier, "%d", &ntx) != 1)
    ntx = 12;
  if (fscanf(fichier, "%d", &nty) != 1)
    nty = 10;
  fclose(fichier);

  u = malloc(ntx*nty*sizeof(double));
  for (iter=0; iter<ntx*nty; iter++)
    u[iter] = 0.;

  code = MPI_File_open(MPI_COMM_WORLD, "donnees.dat", MPI_MODE_RDONLY, 
   MPI_INFO_NULL, &descripteur);
  if (code != MPI_SUCCESS) {
    MPI_Error_string(code, texte_erreur, &texte_longueur);
    printf("%s\n", texte_erreur);
    MPI_Abort(MPI_COMM_WORLD,42);
  }

  MPI_File_set_errhandler(MPI_FILE_NULL,MPI_ERRORS_ARE_FATAL);
  MPI_File_get_size(descripteur, &taille_fichier);
  MPI_Type_size(MPI_DOUBLE, &taille_reel);
  if (taille_fichier != ntx*nty*taille_reel) {
    printf("ATTENTION Donnees.dat n'a pas la bonne longueur\n");
    printf("Taille du fichier : %lld\n", taille_fichier);
    printf("Taille attendue : %d\n", ntx*nty*taille_reel);
  } else {
    fichier = fopen("fort.11", "w");
    MPI_File_read(descripteur, u, ntx*nty, MPI_DOUBLE, &statut);
    for (iter=0; iter<ntx*nty; iter++) {
      fprintf(fichier, "%12.5e\n", u[iter]); }
    fclose(fichier);
  }

  MPI_File_close(&descripteur);

  free(u);

  MPI_Finalize();
  return 0;
}
