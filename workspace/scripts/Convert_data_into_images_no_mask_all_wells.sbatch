#!/bin/bash

#SBATCH -o ./SLURM_output/Convert_data_into_images_all_wells_no_mask.out
#SBATCH -e ./SLURM_output/Convert_data_into_images_all_wells_no_mask_error.out
#SBATCH -J Conv2images_no_mask
#SBATCH -p icb_cpu
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64],icb-rsrv[05-06,08]
#SBATCH -c 3
#SBATCH --mem-per-cpu 64G
#SBATCH -t 15:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/Transform_MPPData_into_images_2_from_script.ipynb -p ./Parameters/Convert_data_into_images_all_wells_no_mask.json
