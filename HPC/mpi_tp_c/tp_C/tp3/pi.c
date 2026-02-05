#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {
  long long nbbloc,i;
  double largeur, somme, x;

  /* Nombre d'intervalles */
  nbbloc = 3*1000*1000LL*100;
  /* largeur des intervalles */
  largeur = 1.0/nbbloc;

  somme = 0;

  for (i=0; i<nbbloc; i++) {
    /* Point au milieu de l'intervalle */
    x = largeur*(i+0.5);
    /* Calcul de l'aire */
    somme = somme + largeur*(4.0 / (1.0 + x*x));
  }

  printf("Pi = %.12lf\n", somme);
  printf("Ecart = %g\n",somme-4.0*atan(1.0));
  
  return 0;
}