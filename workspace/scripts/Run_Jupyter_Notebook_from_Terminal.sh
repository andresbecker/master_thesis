#! /bin/bash
################################################################################
#     This script executs Jupyter notebooks from terminal
#     given an input jupyter notebook and its json file
#     containing its parameters and vars.
#     The output jupither notebook will be located in the
#     same directory as the input notebook plus /NB_output/
# Created by Andres Becker
################################################################################

# Check if needed number of parameters were given
if [ $# -ne 4 ]; then
  echo -e "\nPlease specify parameters file and input notebook.\nFor example:"
  echo -e $0" -p /absolute_path_to_file/parameters.json -i /absolute_path_to_file/notebook.ipynb\n"
  exit 1
fi

PARM_FILE=''
INPUT_NOTEBOOK=''

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
      echo -e $0" -p /absolute_path_to_file/parameters.json -i /absolute_path_to_file/notebook.ipynb\n"
      exit 1
      ;;
  esac
done

echo -e "\nExecution started at:"
date

# Convert relative paths to absolute paths
PARM_FILE=$(readlink -m $PARM_FILE)
INPUT_NOTEBOOK=$(readlink -m $INPUT_NOTEBOOK)

# Check if parameters file exist
if [ ! -f "$PARM_FILE" ]; then
  echo -e "\nError! Parameters file "$PARM_FILE" does not exist!"
  exit 1
fi
echo -e "\nParameters file:\n"$PARM_FILE

# Check if input notebook exist
if [ ! -f "$INPUT_NOTEBOOK" ]; then
  echo -e "\nError! Input notebook "$INPUT_NOTEBOOK" does not exist!"
  exit 1
fi
echo -e "\nInput notebook file:\n"$INPUT_NOTEBOOK

# Create notebook (NB) output dir if it does not exist yet
#SCRIPT=$(realpath $0)
NBs_PATH=$(dirname $INPUT_NOTEBOOK)
PAR_PATH=$(dirname $PARM_FILE)
OUTPUT_PATH=$(readlink -m  $NBs_PATH"/NB_output/")
if [ ! -d $OUTPUT_PATH ]; then
  echo -e "\nCreating notebook output dir "$OUTPUT_PATH"\n"
  mkdir $OUTPUT_PATH
fi

# Create output notebook
# First create output notebook absolute path
TIME_TAG=$(date +"%d%m%y_%H%M")
OUTPUT_NOTEBOOK=$(basename $INPUT_NOTEBOOK | awk 'BEGIN{FS="."}{print $1}')
OUTPUT_NOTEBOOK=$OUTPUT_PATH"/"$OUTPUT_NOTEBOOK"_"$TIME_TAG".ipynb"
# Now, replace input parametrs file with correct one
awk -v parm_file="$PARM_FILE" '{
  # include input variables in the output jupyter notebook
  gsub("dont_touch_me-input_parameters_file", parm_file);
  # print new notebook
  print $0}' $INPUT_NOTEBOOK > $OUTPUT_NOTEBOOK

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

# Execute notebook
jupyter nbconvert --to notebook --execute $OUTPUT_NOTEBOOK --inplace --allow-errors
#jupyter-nbconvert --to notebook --execute $INPUT_NOTEBOOK --allow-errors --output $OUTPUT_NAME
#jupyter-nbconvert --to notebook --execute $INPUT_NOTEBOOK --allow-errors --output-dir $OUTPUT_PATH

echo -e "\nOutput notbook:"
echo $OUTPUT_NOTEBOOK

echo -e "\nExecution ended at:"
date

echo -e "\nScript Finished"
