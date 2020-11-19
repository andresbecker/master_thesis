#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Project_data_to_scaler.out
#SBATCH -e /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Project_data_to_scaler_errors.out
#SBATCH -J Project2scalars
#SBATCH -p icb_cpu
#SBATCH -c 1
#SBATCH --mem-per-cpu 32G
#SBATCH -t 12:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/notebooks/Data_Preprocessing_only_scalars_from_script.ipynb -p /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/Parameters/params_to_scalar_only.json
