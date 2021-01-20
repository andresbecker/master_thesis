#!/bin/bash

#SBATCH -o ./SLURM_output/BL.log
#SBATCH -e ./SLURM_output/BL.error.log
#SBATCH -J BL
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 16:00:00
#SBATCH --nice=10000

../Run_Jupyter_Notebook_from_Terminal.sh -i ../../notebooks/CNN_model_from_script3.ipynb -p ./Parameters/BL.json
