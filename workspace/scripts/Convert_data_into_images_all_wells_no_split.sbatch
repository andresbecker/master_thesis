#!/bin/bash

#SBATCH -o ./SLURM_output/Convert_data_imgs_all_wels_no_split.log
#SBATCH -e ./SLURM_output/Convert_data_imgs_all_wels_no_split.error.log
#SBATCH -J Conv2images
#SBATCH -p cpu_p
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64],icb-rsrv[05-06,08]
#SBATCH -c 3
#SBATCH --mem-per-cpu 64G
#SBATCH -t 15:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/MPPData_into_images_no_split.ipynb -p ./Parameters/MppData_to_imgs_no_split.json
