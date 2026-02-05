############################ -*- Mode: Makefile -*- ###########################
## Makefile --- Cours MPI : TP1 : Environnement
##
## Auteur          : Dimitri Lecas (CNRS/IDRIS) 
##
###############################################################################

# Compilateurs, options de compilation et d'édition de liens
include ../../arch/make_inc

OBJS = pair_impair.o

# Règle implicite de compilation
.SUFFIXES: .o .c
%.o : %.c
	$(CC) -c $(CFLAGS_TP1) $<

all: pair_impair

pair_impair:$(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS_TP1)

.PHONY: clean exe

exe : pair_impair
	$(MPIEXEC_TP1) ./pair_impair
#SBATCH	sbatch pair_impair.slurm

clean:
	rm -f $(OBJS) pair_impair core
