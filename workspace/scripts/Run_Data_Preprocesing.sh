#! /bin/bash

source ~/.bashrc
source activate icb_mt

SCRIPT=$(realpath $0)
SCRIPT_PATH=$(dirname $SCRIPT)
NBs_PATH=$SCRIPT_PATH"/../notebooks/"
NB_NAME=Data_Preprocessing
#NB_NAME=Data_Preprocessing.ipynb
NOTEBOOK=$NBs_PATH$NB_NAME

# Create notebook (NB) output dir if it does not exist yet
if [ ! -d $NBs_PATH"NB_output" ]; then
  mkdir $NBs_PATH"NB_output"
fi

#jupyter-nbconvert --to html --ExecutePreprocessor.enabled=True $NOTEBOOK --output $NBs_PATH"NB_output/"$NB_NAME
jupyter-nbconvert --to html --execute $NOTEBOOK --output $NBs_PATH"NB_output/"$NB_NAME
