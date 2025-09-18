#!/bin/bash
#SBATCH --account=siliconranch
#SBATCH --job-name=pyconturb_40
##SBATCH --time=1:00:00
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --qos=high
##SBATCH --partition=debug
#SBATCH --output=out_%x_%j.log
#SBATCH --error=err_%x_%j.log
#SBATCH --mail-user=brooke.stanislawski@nrel.gov
#SBATCH --mail-type=ALL

module load mamba
mamba activate pyconturb

export OMP_NUM_THREADS=26

srun python /projects/pvopt/brooke/duramat-validation-turbinflow/pyconturb/pyconturb-pvade/hpc_duramat_constrained_turb.py
