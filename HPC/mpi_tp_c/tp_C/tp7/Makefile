############################ -*- Mode: Makefile -*- ###########################
## Makefile --- T.P. 7 du cours MPI
##
## Auteur          : Dimitri Lecas (CNRS/IDRIS - France) 
##
###############################################################################

# Compilateur et options de compilation et de link
include ../arch/make_inc

OBJS    = lire_fichier.o

# Règle implicite de compilation
.SUFFIXES:
.SUFFIXES: .o .c
%.o : %.c
	$(CC) -c $(CFLAGS_TP7) $<

lire_fichier:donnees.dat $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS_TP7) -o $@

donnees.dat:
	cd creation;make clean;make

verification: verification.c fichier_dei0.dat
	$(CC) verification.c -o verification
	$(EXEC_TP7_VERIF) ./verification
	md5sum -c checkmd5.md5

.PHONY : clean cleanall exe

exe: lire_fichier
	$(MPIEXEC_TP7) ./lire_fichier
#SBATCH	sbatch lire_fichier.slurm

clean:
	rm -f *.o lire_fichier fichier*.dat donnees.dat *.ppm verification core*

cleanall:
	make clean
	rm -f donnees.dat
