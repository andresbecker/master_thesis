#!/bin/bash

#SBATCH -o ./SLURM_output/Pretrained_ResNet50V2_wo_pert.log
#SBATCH -e ./SLURM_output/Pretrained_ResNet50V2_wo_pert.error.log
#SBATCH -J PreTrain_ResNet50V2_wo_pert
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5],supergpu02pxe,supergpu03pxe
#SBATCH -c 6
#SBATCH --mem-per-cpu 8G
#SBATCH -t 16:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/PreTrained_Resnet50V2_from_script.ipynb -p ./Parameters/model_params/Pretrained_ResNet50V2_wo_perturbations.json
