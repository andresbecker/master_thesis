#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Convert_data_into_images_all_wells.out
#SBATCH -e /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Convert_data_into_images_all_wells_errors.out
#SBATCH -J Conv2images
#SBATCH -p icb_cpu
#SBATCH -c 2
#SBATCH --mem-per-cpu 64G
#SBATCH -t 12:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/notebooks/Data_Preprocessing_from_script.ipynb -p /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/Parameters/params_imgs_and_scalar.json
