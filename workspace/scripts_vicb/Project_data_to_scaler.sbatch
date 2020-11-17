#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts_vicb/SLURM_output/Project_data_to_scaler.out
#SBATCH -e /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts_vicb/SLURM_output/Project_data_to_scaler_errors.out
#SBATCH -J Project2scalars
#SBATCH -p icb_cpu
#SBATCH -c 1
#SBATCH --mem-per-cpu 64G
#SBATCH -t 3:00:00
#SBATCH --nice=10000

./Run_Data_Preprocesing.sh -p /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts_vicb/Parameters/params_to_scalar_only.json -i /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/notebooks_vicb/Data_Preprocessing_only_projection.ipynb
