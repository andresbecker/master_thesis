#!/bin/bash

#SBATCH -o ./SLURM_output/Pretrained_ResNet50V2.out
#SBATCH -e ./SLURM_output/Pretrained_ResNet50V2_error.out
#SBATCH -J PreTrain_ResNet50V2
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5]
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 8:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/PreTrained_Resnet50V2_from_script.ipynb -p ./Parameters/model_params/Pretrained_ResNet50V2.json
