#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Train_baseline_CNN_wo_augm.out
#SBATCH -e /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/SLURM_output/Train_baseline_CNN_wo_augm_error.out
#SBATCH -J BaseLine_w_a
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5]
#SBATCH -c 3
#SBATCH --mem-per-cpu 16G
#SBATCH -t 3:00:00
#SBATCH --nice=10000


./Run_Jupyter_Notebook_from_Terminal.sh -i /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/notebooks/CNN_model_from_script.ipynb -p /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/scripts/Parameters/model_params/params_model_baseline_wo_ougmentation.json
