#!/bin/bash
################################################################################
#                        Script to create tf datasets
#   This script needs to be located in the same dir where the dirs containing
#   the td datasets builders are.
#   The inputs are the name of the dataset (the name of the dir where the
#   builder is), the absolute path where the output tf dataset will be
#   located and the absolute path to the input parameters (json file) that
#   contains the information to create the tf dataset.
# Created by Andres Becker
################################################################################

# Check if needed number of parameters were given
if [ $# -ne 6 ]; then
  echo -e "\nPlease specify the tf dataset name, the output directory and the path to the input parameters.\nFor example:"
  echo -e $0" -o /TFDS_DATA_DIR -n MPP_dataset -p /absolute_path/params.json\n"
  exit 1
fi

TF_DATASET_NAME=""
TFDS_DATA_DIR=""
PARM_FILE=""
#TF_DATASET_NAME="MPP_dataset"
#TFDS_DATA_DIR="/home/hhughes/Documents/Master_Thesis/Project/datasets/tensorflow_datasets"


# Read parameters and values
while getopts 'o:n:p:' flag; do
  case "${flag}" in
    o)
      TFDS_DATA_DIR="${OPTARG}"
      ;;
    n)
      TF_DATASET_NAME="${OPTARG}"
      ;;
    p)
      PARM_FILE="${OPTARG}"
      ;;
    *)
      echo -e "\nPlease specify the tf dataset name, the output directory and the path to the input parameters.\nFor example:"
      echo -e $0" -o /TFDS_DATA_DIR -n MPP_dataset -i /absolute_path/params.json\n"
      exit 1
      ;;
  esac
done

# Check if parameters file exist
PARM_FILE=$(readlink -m $PARM_FILE)
if [ ! -f "$PARM_FILE" ]; then
  echo -e "\nError! Parameters file "$PARM_FILE" does not exist!"
  exit 1
fi
echo -e "\nParameters file:\n"$PARM_FILE

# Get params dir
PARM_dir=$(dirname $PARM_FILE)

# Create parameter file
cat $PARM_FILE > $PARM_dir"/tf_dataset_parameters.json"

# Get the absolute path where the tf dataset builders are
DATASET_BUILDER_PATH=$(realpath $0)
DATASET_BUILDER_PATH=$(dirname $DATASET_BUILDER_PATH)
#SCRIPT_PATH=$DATASET_BUILDER_PATH
DATASET_BUILDER_PATH=$(readlink -m $DATASET_BUILDER_PATH"/"$TF_DATASET_NAME)
cd $DATASET_BUILDER_PATH

# Create the tf datasets ouput dir
if [ ! -d $TFDS_DATA_DIR ]; then
  echo -e "\nCreating tf datasets dir "$TFDS_DATA_DIR"\n"
  mkdir $TFDS_DATA_DIR
fi

# Load conda environment
source ~/.bashrc
ENVI="icb_mt"
echo -e "\nLoading environment "$ENVI"..."
source activate $ENVI
if [ $? -eq 0 ]; then
  echo -e "Environment "$ENVI" loaded successfully!\n"
else
  echo -e "\Error while loading environment "$ENVI"!!!!\n"
  exit 1
fi

# Export variable to change default tf dataset output
export TFDS_DATA_DIR

# To run the following command install first (in your conda environment):
# pip install -q tfds-nightly
#tfds build --register_checksums 2>&1 | tee  $SCRIPT_PATH"/Create_tf_dataset.log"
tfds build --register_checksums

echo "Execution of "$0" finished."
