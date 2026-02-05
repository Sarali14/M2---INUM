############################ -*- Mode: Makefile -*- ###########################
## Makefile --- Cours MPI : TP5 : Communications collectives
##
## Auteur          : Dimitri LECAS (CNRS/IDRIS)
##
###############################################################################

# Compilateurs, options de compilation et d'édition de liens
include ../arch/make_inc

OBJS = produit_matrices.o

# Règle implicite de compilation
.SUFFIXES: .o .c
%.o : %.c
	$(CC) -c $(CFLAGS_TP5) $<

all : produit_matrices

produit_matrices: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS_TP5)

.PHONY: clean exe

exe: produit_matrices
	$(MPIEXEC_TP5) ./produit_matrices
#SBATCH	sbatch produit_matrices.slurm

clean:
	rm -f $(OBJS) produit_matrices core
