#!/bin/bash

#SBATCH -o ./SLURM_output/%x.log
#SBATCH -e ./SLURM_output/%x.error.log
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5]
#SBATCH -c 4
#SBATCH --mem-per-cpu 12G
#SBATCH -t 2-00:00:00
#SBATCH --nice=10000

../Run_Jupyter_Notebook_from_Terminal.sh -i ../../notebooks/Model_training_class.ipynb -p ./Parameters/$PARAMS_DIR/$PARAMS.json
