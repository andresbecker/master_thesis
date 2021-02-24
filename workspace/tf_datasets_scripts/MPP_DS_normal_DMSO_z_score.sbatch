#!/bin/bash

#SBATCH -o ./SLURM_output/MPP_DS_normal_DMSO_z_score.log
#SBATCH -e ./SLURM_output/MPP_DS_normal_DMSO_z_score.error.log
#SBATCH -J Create_tf_dataset
#SBATCH -p cpu_p
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64,02],icb-rsrv[05-06,08]
#SBATCH -c 4
#SBATCH --mem-per-cpu 32G
#SBATCH -t 23:59:00
#SBATCH --nice=10000

./Create_tf_dataset.sh -o /storage/groups/ml01/workspace/andres.becker/datasets/tensorflow_datasets -n MPP_DS_normal_DMSO_z_score -p ./MPP_DS_normal_DMSO_z_score/Parameters/tf_dataset_parameters_server.json -e icb_mt
