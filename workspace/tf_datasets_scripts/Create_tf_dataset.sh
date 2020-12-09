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

ERROR_msg="\nPlease specify the tf dataset name, the output directory, the path where the input parameters file is and the conda environment name where the tf dataset will be registered.\nFor example:\n"$0" -n MPP_dataset -o /TFDS_DATA_DIR -p /absolute_path/params.json -e my_conda_env_name\n"

# Check if needed number of parameters were given
if [ $# -ne 8 ]; then
  echo -e $ERROR_msg
  exit 1
fi

TF_DATASET_NAME=""
TFDS_DATA_DIR=""
PARM_FILE=""
ENVI=""
#TF_DATASET_NAME="MPP_dataset"
#TFDS_DATA_DIR="/home/hhughes/Documents/Master_Thesis/Project/datasets/tensorflow_datasets"


# Read parameters and values
while getopts 'o:n:p:e:' flag; do
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
    e)
      ENVI="${OPTARG}"
      ;;
    *)
      echo -e $ERROR_msg
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

# Install the tensorflow datasets builder in the specified conda environment:
pip install -q tfds-nightly
#tfds build --register_checksums 2>&1 | tee  $SCRIPT_PATH"/Create_tf_dataset.log"
tfds build --register_checksums
OUT_FLAG=$?

echo "Execution of "$0" finished."

exit $OUT_FLAG
