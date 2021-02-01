#!/bin/bash
################################################################################
#                       Script to execut python script in server
#   The inputs are the path to the python script, the path where the input
#   parameters file (json) is and the name of the conda environment.
# Created by Andres Becker
################################################################################

ERROR_msg="\nPlease specify the conda environment name, the path to the python script and the path where the input parameters file is.\nFor example:\n"$0" -e my_conda_env_name -s /path_to_file/python_script.py -p /absolute_path/params.json\n"

# Check if needed number of parameters were given
if [ $# -ne 6 ]; then
  echo -e $ERROR_msg
  exit 1
fi

PYTHON_SCRIPT=""
PARM_FILE=""
ENVI=""


# Read parameters and values
while getopts 's:p:e:' flag; do
  case "${flag}" in
    s)
      PYTHON_SCRIPT="${OPTARG}"
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

# Check if python script file exist
PYTHON_SCRIPT=$(readlink -m $PYTHON_SCRIPT)
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo -e "\nError! Python script file "$PYTHON_SCRIPT" does not exist!"
  exit 1
fi
echo -e "\nPython scriipt file:\n"$PYTHON_SCRIPT

# Check if parameters file exist
PARM_FILE=$(readlink -m $PARM_FILE)
if [ ! -f "$PARM_FILE" ]; then
  echo -e "\nError! Parameters file "$PARM_FILE" does not exist!"
  exit 1
fi
echo -e "\nParameters file:\n"$PARM_FILE


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

# Execute python script
python $PYTHON_SCRIPT -i $PARM_FILE --cell_id "321021"

echo "Execution of "$0" finished."
