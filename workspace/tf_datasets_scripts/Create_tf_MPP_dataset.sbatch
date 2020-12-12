#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/tf_datasets_scripts/SLURM_output/Create_MPP_dataset.out
#SBATCH -e /storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/tf_datasets_scripts/SLURM_output/Create_MPP_dataset_errors.out
#SBATCH -J Create_tf_dataset
#SBATCH -p icb_cpu
#SBATCH -c 4
#SBATCH --mem-per-cpu 32G
#SBATCH -t 23:59:00
#SBATCH --nice=10000

./Create_tf_dataset.sh -o /storage/groups/ml01/workspace/andres.becker/datasets/tensorflow_datasets -n MPP_dataset -p ./MPP_dataset/Parameters/tf_dataset_parameters_server.json -e icb_mt
