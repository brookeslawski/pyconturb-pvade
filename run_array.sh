#!/bin/bash
#SBATCH --account=siliconranch
#SBATCH --job-name=pyconturb
##SBATCH --time=1:00:00
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --qos=high
##SBATCH --partition=debug
#SBATCH --output=genturb_%A_%a.out
#SBATCH --error=genturb_%x_%j.err
#SBATCH --mail-user=brooke.stanislawski@nrel.gov
#SBATCH --mail-type=ALL

module load mamba
mamba activate pyconturb

export OMP_NUM_THREADS=26

# Define all the csv filenames
gen_csv_fname=(
    "generated_DuraMAT_tilt10deg_turbulent_inflow_30s_50Hz.csv"
    "generated_DuraMAT_tiltneg10deg_turbulent_inf_30s_50Hz.csv"
    "generated_DuraMAT_tilt40deg_turbulent_inflow_30s_50Hz.csv"
)

# Select the casepath based on SLURM_ARRAY_TASK_ID
gen_csv_fname="${gen_csv_fname[$SLURM_ARRAY_TASK_ID]}"

# Run your script for each casepath
srun python /projects/pvopt/brooke/duramat-validation-turbinflow/pyconturb/pyconturb-pvade/hpc_duramat_constrained_turb.py --gen_csv_fname "$gen_csv_fname"
