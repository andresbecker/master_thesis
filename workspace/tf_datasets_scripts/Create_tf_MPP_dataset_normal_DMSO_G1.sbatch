#!/bin/bash

#SBATCH -o ./SLURM_output/Create_tf_MPP_dataset_normal_DMSO_G1.log
#SBATCH -e ./SLURM_output/Create_tf_MPP_dataset_normal_DMSO_G1.error.log
#SBATCH -J Create_tf_dataset
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem-per-cpu 32G
#SBATCH -t 23:59:00
#SBATCH --nice=10000

./Create_tf_dataset.sh -o /storage/groups/ml01/workspace/andres.becker/datasets/tensorflow_datasets -n MPP_dataset_Normal_DMSO_G1 -p ./MPP_dataset_Normal_DMSO_G1/Parameters/tf_dataset_parameters_server.json -e icb_mt
