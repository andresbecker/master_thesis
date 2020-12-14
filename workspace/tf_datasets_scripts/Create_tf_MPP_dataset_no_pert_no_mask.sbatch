#!/bin/bash

#SBATCH -o ./SLURM_output/MPP_dataset_no_pert_no_mask.out
#SBATCH -e ./SLURM_output/MPP_dataset_no_pert_no_mask_error.out
#SBATCH -J Create_tf_dataset
#SBATCH -p icb_cpu
#SBATCH -c 4
#SBATCH --mem-per-cpu 32G
#SBATCH -t 23:59:00
#SBATCH --nice=10000

./Create_tf_dataset.sh -o /storage/groups/ml01/workspace/andres.becker/datasets/tensorflow_datasets -n MPP_dataset_no_pert_no_mask -p ./MPP_dataset_no_pert_no_mask/Parameters/tf_dataset_parameters_server.json -e icb_mt
