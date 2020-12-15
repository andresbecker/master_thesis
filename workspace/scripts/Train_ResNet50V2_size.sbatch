#!/bin/bash

#SBATCH -o ./SLURM_output/ResNet50V2_size.log
#SBATCH -e ./SLURM_output/ResNet50V2_size.erro.log
#SBATCH -J ResNet50V2_size
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5],icb-gpusrv0[6-8]
#SBATCH -c 4
#SBATCH --mem-per-cpu 16G
#SBATCH -t 8:00:00
#SBATCH --nice=10000

./Run_Jupyter_Notebook_from_Terminal.sh -i ../notebooks/CNN_model_size_from_script.ipynb -p ./Parameters/model_params/ResNet50V2_size.json
