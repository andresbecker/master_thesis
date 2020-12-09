###############################################################################
# This script register the tfds MPP image dataset in the given conda
# environment. The tfds MPP image dataset was already builded, however if you
# want to use it with a conda environment different from the one used to build
# it, then you need to register it first and this script make it for you.
# The only input forthis script is the name of your conda env
# (-e conda_env_name).
###############################################################################

# Predefined values
TF_DATASET_NAME="MPP_dataset"
TFDS_DATA_DIR="/storage/groups/ml01/workspace/andres.becker/master_thesis/datasets/tensorflow_datasets"
PARM_FILE="/storage/groups/ml01/workspace/andres.becker/master_thesis/workspace/tf_datasets_scripts/MPP_dataset/Parameters/tf_dataset_parameters_server.json"
ENVI=""

ERROR_msg="\nPlease specify the the conda environment name where the tf dataset will be registered.\nFor example:\n"$0" -e my_conda_env_name\n"

# Check if needed number of parameters were given
if [ $# -ne 2 ]; then
  echo -e $ERROR_msg
  exit 1
fi

ENVI=""
#TF_DATASET_NAME="MPP_dataset"
#TFDS_DATA_DIR="/home/hhughes/Documents/Master_Thesis/Project/datasets/tensorflow_datasets"

# Read parameters and values
while getopts 'e:' flag; do
  case "${flag}" in
    e)
      ENVI="${OPTARG}"
      ;;
    *)
      echo -e $ERROR_msg
      exit 1
      ;;
  esac
done

#./Create_tf_dataset.sh -o /home/hhughes/Documents/Master_Thesis/Project/datasets/tensorflow_datasets -n MPP_dataset -p MPP_dataset/Parameters/tf_dataset_parameters_local.json -e $ENVI

# For the vicb servers and for anybody who wnats to use the MPP images datadet!
./Create_tf_dataset.sh -o $TFDS_DATA_DIR -n $TF_DATASET_NAME -p $PARM_FILE -e $ENVI

OUT=$?
if [ $OUT -eq 0 ]; then
  echo "MPP image dataset registration successfull!"
  echo -e "\nTensorflow dataset located in:\n"$TFDS_DATA_DIR
  echo -e "\nTo use it add:\ntfds.load(name='MPP_dataset', data_dir='"$TFDS_DATA_DIR"') to your python script!\n"
  exit 0
else
  echo "Error! ./Create_tf_dataset.sh output: "$OUT$" different from 0!"
  exit 1
fi
