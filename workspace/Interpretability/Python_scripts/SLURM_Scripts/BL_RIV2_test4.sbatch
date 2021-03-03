#!/bin/bash

#SBATCH -o ./SLURM_output/BL_RIV2_test4.log
#SBATCH -e ./SLURM_output/BL_RIV2_test4.error.log
#SBATCH -J Score_BL_RIV2_test4
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2],icb-gpusrv03,icb-gpusrv0[4-5]
#SBATCH -c 4
#SBATCH --mem-per-cpu 12G
#SBATCH -t 24:00:00
#SBATCH --nice=10000

../Run_pyhton_script.sh -e icb_mt -s ../get_VarGradIG_from_TFDS.py -p ../Parameters/BL_RIV2_test4.json
