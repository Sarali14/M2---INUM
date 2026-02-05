############################ -*- Mode: Makefile -*- ###########################
## Makefile --- Cours MPI : TP6 : Subdiviser une grille 2D avec MPI_COMM_SPLIT
##
## Auteur          : Dimitri Lecas (CNRS/IDRIS)
##
###############################################################################

# Compilateurs, options de compilation et d'édition de liens
include ../arch/make_inc

OBJS = commsplit.o

# Règle implicite de compilation
.SUFFIXES: .o .c
%.o : %.c
	$(CC) -c $(CFLAGS_TP6) $<

all : commsplit

commsplit: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS_TP6)

.PHONY: clean exe

exe: commsplit
	$(MPIEXEC_TP6) ./commsplit
#SBATCH	sbatch commsplit.slurm

clean :
	rm -f $(OBJS) commsplit core
