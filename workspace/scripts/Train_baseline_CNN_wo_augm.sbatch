#!/bin/bash

#SBATCH -o ./SLURM_output/Train_baseline_CNN_wo_augm.out
#SBATCH -e ./SLURM_output/Train_baseline_CNN_wo_augm_error.out
#SBATCH -J BaseLine_wo_a
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5]
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 5:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/CNN_model_from_script.ipynb -p ./Parameters/model_params/Train_baseline_CNN_wo_augm.json
