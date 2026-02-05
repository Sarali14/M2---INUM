############################ -*- Mode: Makefile -*- ###########################
## Makefile --- Cours MPI : TP4 : Type de données derivées (type_transpose)
##
## Auteur          : Dimitri LECAS (CNRS/IDRIS) <dimitri.lecas@idris.fr>
##
###############################################################################

# Compilateurs, options de compilation et d'édition de liens
include ../arch/make_inc

OBJS = transpose.o

# Règle implicite de compilation
.SUFFIXES: .o .c
%.o : %.c
	$(CC) -c $(CFLAGS_TP4) $<

all : transpose

transpose: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS_TP4)

.PHONY: clean exe

exe : transpose
	$(MPIEXEC_TP4) ./transpose
#SBATCH	sbatch transpose.slurm

clean :
	rm -f $(OBJS) transpose core
