#!/bin/bash

#SBATCH -o ./SLURM_output/BL_test4.log
#SBATCH -e ./SLURM_output/BL_test4.error.log
#SBATCH -J BL_test4
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 16:00:00
#SBATCH --nice=10000

../Run_Jupyter_Notebook_from_Terminal.sh -i ../../notebooks/CNN_model_from_script2.ipynb -p ./Parameters/BL_test4.json
