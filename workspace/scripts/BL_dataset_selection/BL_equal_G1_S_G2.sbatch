#!/bin/bash

#SBATCH -o ./SLURM_output/BL_equal_G1_S_G2.log
#SBATCH -e ./SLURM_output/BL_equal_G1_S_G2.error.log
#SBATCH -J BL_equal_G1_S_G2
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv03,icb-gpusrv05,icb-gpusrv06,icb-gpusrv07,icb-gpusrv08,supergpu02pxe,supergpu03pxe
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 16:00:00
#SBATCH --nice=10000

../Run_Jupyter_Notebook_from_Terminal.sh -i ../../notebooks/CNN_model_from_script2.ipynb -p ./Parameters/BL_equal_G1_S_G2.json
