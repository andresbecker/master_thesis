#! /bin/bash

PARM_FILE=''
INPUT_NOTEBOOK=''

if [ $# -ne 2 ]; then
  echo -e "Please specify parameters file and input notebook.\nFor example:"
  echo $0" -p /path_to_file/parameters.json -i /path_to_file/notebook.ipynb"
  exit 1
fi

while getopts 'p:i:' flag; do
  case "${flag}" in
    p)
      PARM_FILE="${OPTARG}"
      ;;
    i)
      INPUT_NOTEBOOK="${OPTARG}"
      ;;
    *)
      echo "Flag "$flag" is not recognized!"
      exit 1
      ;;
  esac
done

echo $PARM_FILE
echo $INPUT_NOTEBOOK

echo -e "\nFinishing script"
