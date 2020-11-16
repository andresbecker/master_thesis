#! /bin/bash

source ~/.bashrc
ENVI="icb_mt"
echo "Loading environment "$ENVI
source activate $ENVI
if [ $? -eq 0 ]; then
  echo -e "Environment "$ENVI" loaded successfully\n"
fi

SCRIPT=$(realpath $0)
SCRIPT_PATH=$(dirname $SCRIPT)
NBs_PATH=$SCRIPT_PATH"/../notebooks_vicb/"
NB_NAME=Data_Preprocessing_from_script
NOTEBOOK=$NBs_PATH$NB_NAME

# Create notebook (NB) output dir if it does not exist yet
if [ ! -d $NBs_PATH"NB_output" ]; then
  echo -e "Creating notebook output dir "$NBs_PATH"NB_output\n"
  mkdir $NBs_PATH"NB_output"
fi

# Create temporary file that contains the input parameters
#echo -e "Creating temporary file containing the parameters for notebook"
#PARM_FILE='params_I09.json'
#TEMP_PARM_FILE='temp_parameters.json'
#rsync -avz $SCRIPT_PATH"/Parameters/"$PARM_FILE $SCRIPT_PATH"/Parameters/"$TEMP_PARM_FILE

echo -e "Executing notebook...\n"
#jupyter-nbconvert --to html --execute $NOTEBOOK --output $NBs_PATH"NB_output/"$NB_NAME
#jupyter-nbconvert --to notebook --execute $NOTEBOOK --allow-errors --output $NBs_PATH"NB_output/"$NB_NAME"_processed"

#rm $SCRIPT_PATH"/Parameters/"$TEMP_PARM_FILE
