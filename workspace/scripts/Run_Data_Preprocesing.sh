#! /bin/bash
################################################################################
#     This script executs Jupyter notebooks from terminal
#     given an input jupyter notebook and its json file
#     containing its parameters.
################################################################################

echo -e "\nExecution started at:"
date

PARM_FILE=''
INPUT_NOTEBOOK=''

# Check if needed number of parameters were given
if [ $# -ne 4 ]; then
  echo -e "\nPlease specify parameters file and input notebook.\nFor example:"
  echo $0" -p /absolute_path_to_file/parameters.json -i /absolute_path_to_file/notebook.ipynb"
  exit 1
fi

# Read parameters and values
while getopts 'p:i:' flag; do
  case "${flag}" in
    p)
      PARM_FILE="${OPTARG}"
      ;;
    i)
      INPUT_NOTEBOOK="${OPTARG}"
      ;;
    *)
      echo -e "\nPlease specify parameters file and input notebook.\nFor example:"
      echo $0" -p /path_to_file/parameters.json -i /path_to_file/notebook.ipynb"
      exit 1
      ;;
  esac
done

# Convert relative paths to absolute paths
PARM_FILE=$(readlink -m $PARM_FILE)
INPUT_NOTEBOOK=$(readlink -m $INPUT_NOTEBOOK)

# Check if given files exist
if [ ! -f "$PARM_FILE" ]; then
  echo -e "\nError! Parameters file "$PARM_FILE" does not exist!"
  exit 1
fi
echo -e "\nParameters file:\n"$PARM_FILE

# Check if given files exist
if [ ! -f "$INPUT_NOTEBOOK" ]; then
  echo -e "\nError! Input notebook "$INPUT_NOTEBOOK" does not exist!"
  exit 1
fi
echo -e "\nInput notebook file:\n"$INPUT_NOTEBOOK

# Load conda environment
source ~/.bashrc
ENVI="icb_mt"
echo -e "\nLoading environment "$ENVI
source activate $ENVI
if [ $? -eq 0 ]; then
  echo -e "\nEnvironment "$ENVI" loaded successfully\n"
else
  echo -e "\nError while loading environment "$ENVI"!\n"
  exit 1
fi

#SCRIPT=$(realpath $0)
#SCRIPT_PATH=$(dirname $SCRIPT)
NBs_PATH=$(dirname $INPUT_NOTEBOOK)
PAR_PATH=$(dirname $PARM_FILE)
OUTPUT_PATH=$(readlink -m  $NBs_PATH"/NB_output/")

# Create notebook (NB) output dir if it does not exist yet
if [ ! -d $OUTPUT_PATH ]; then
  echo -e "\nCreating notebook output dir "$OUTPUT_PATH"\n"
  mkdir $OUTPUT_PATH
fi

# Create temporary file that contains the input parameters
echo -e "\nCreating temporary file containing the parameters for notebook"
TEMP_PARM_FILE=$PAR_PATH"/temp_parameters.json"
cat $PARM_FILE > $TEMP_PARM_FILE

# Create output name
TIME_TAG=$(date +"%d%m%y_%H%M")
OUTPUT_NAME=$(basename $INPUT_NOTEBOOK | awk 'BEGIN{FS="."}{print $1}')
OUTPUT_NAME=$OUTPUT_PATH"/"$OUTPUT_NAME"_"$TIME_TAG

# Execute notebook
jupyter-nbconvert --to notebook --execute $INPUT_NOTEBOOK --allow-errors --output $OUTPUT_NAME
#jupyter-nbconvert --to notebook --execute $INPUT_NOTEBOOK --allow-errors --output-dir $OUTPUT_PATH

echo -e "\nOutput notbook:"
echo $OUTPUT_NAME

echo -e "\nExecution ended at:"
date

echo -e "\nScript Finished"
