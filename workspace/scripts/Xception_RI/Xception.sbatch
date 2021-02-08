#!/bin/bash

#SBATCH -o ./SLURM_output/XC_RI.log
#SBATCH -e ./SLURM_output/XC_RI.error.log
#SBATCH -J XC_RI
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5],icb-gpusrv0[6-8]
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 2-00:00:00
#SBATCH --nice=10000

../Run_Jupyter_Notebook_from_Terminal.sh -i ../../notebooks/CNN_model_from_script2.ipynb -p ./Parameters/XC_RI.json
