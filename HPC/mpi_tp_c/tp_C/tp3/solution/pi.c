#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

int main(int argc, char *argv[]) {
  long long nbbloc,i;
  double largeur, somme, x;
  int rang,nbprocs;
  long long debut,fin;
  double global;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rang);
  MPI_Comm_size(MPI_COMM_WORLD,&nbprocs);

  /* Nombre d'intervalles */
  nbbloc = 3*1000*1000LL*100;
  /* largeur des intervalles */
  largeur = 1.0/nbbloc;

  somme = 0;

  /* Distribution repartie (Attention aux overflow) */
  /* rang*nbbloc must be less than 9.10^18 */
  debut = (rang*nbbloc)/nbprocs;
  fin = ((rang+1)*nbbloc)/nbprocs;

  /* Idem */
  /* rang*nbbloc must be less than  9.10^15 */
  /*debut = ((1.0*rang)*nbbloc)/nbprocs;
    fin = ((1.0*(rang+1))*nbbloc)/nbprocs;*/

  /* Le reste est distribue sur les premiers rangs */
  /*debut = rang*(nbbloc/nbprocs)+MIN(rang,nbbloc%nbprocs);
    fin = debut+(nbbloc/nbprocs);
    if (rang < (nbbloc%nbprocs)) fin++;*/

  /* Le reste est distribue sur les derniers rangs */
  /*debut = rang*(nbbloc/nbprocs)+MAX((nbbloc%nbprocs)+rang-nbprocs,0);
    fin = debut+(nbbloc+rang)/nbprocs;*/

  printf("%d debut: %lld fin: %lld delta: %lld\n", rang, debut, fin, fin-debut);

  for (i=debut; i<fin; i++) {
    /* Point au milieu de l'intervalle */
    x = largeur*(i+0.5);
    /* Calcul de l'aire */
    somme = somme + largeur*(4.0 / (1.0 + x*x));
  }

  MPI_Reduce(&somme, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rang ==0) printf("Pi = %.12lf\n", global);
  if (rang ==0) printf("Ecart = %g\n", global-4.0*atan(1.0));
  i = fin-debut;
  MPI_Allreduce(MPI_IN_PLACE,&i,1,MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rang ==0) printf("Nb = %lld\n", i);

  MPI_Finalize();
  return 0;
}
