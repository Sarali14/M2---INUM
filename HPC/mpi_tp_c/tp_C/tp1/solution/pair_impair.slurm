#!/bin/bash
#SBATCH --job-name=pair_impair
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:01:00
#SBATCH --output=pair_impair-%j.out
#SBATCH --error=pair_impair-%j.out
#SBATCH --hint=nomultithread
#RESERVATION

cd ${SLURM_SUBMIT_DIR}
 
module purge
 
module load intel-all/2020.4
 
set -x
 
srun ./pair_impair
