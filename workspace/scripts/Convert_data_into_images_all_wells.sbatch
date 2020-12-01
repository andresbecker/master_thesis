#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Convert_data_into_images_all_wells.out
#SBATCH -e /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Convert_data_into_images_all_wells_errors.out
#SBATCH -J Conv2images
#SBATCH -p icb_cpu
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64],icb-rsrv[05-06,08]
#SBATCH -c 1
#SBATCH --mem-per-cpu 128G
#SBATCH -t 15:00:00
#SBATCH --nice=10000


./Run_Jupyter_Notebook_from_Terminal.sh -i /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/notebooks/Transform_MPPData_into_images_from_script.ipynb -p /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/Parameters/params_imgs_and_scalar.json
