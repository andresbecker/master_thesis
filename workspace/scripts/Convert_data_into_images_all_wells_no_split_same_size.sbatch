#!/bin/bash

#SBATCH -o ./SLURM_output/MppData_to_imgs_no_split_and_same_size.log
#SBATCH -e ./SLURM_output/MppData_to_imgs_no_split_and_same_size.error.log
#SBATCH -J Conv2images
#SBATCH -p cpu_p
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64],icb-rsrv[05-06,08]
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 15:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/MPPData_into_images_no_split_and_same_size.ipynb -p ./Parameters/MppData_to_imgs_no_split_and_same_size.json
