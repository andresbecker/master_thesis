import os
import numpy as np

# --- set local paths ---
for BASE_DIR in [
    os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')),
    # insert other paths to base dir (containing local_data/raw_data dirs) 
]:
    if os.path.exists(BASE_DIR):
        print('Setting BASE_DIR to ' + BASE_DIR)
        break
if not os.path.exists(BASE_DIR):
    print('WARNING: BASE_DIR not found, setting to None')
    BASE_DIR = None
    
DATA_DIR = os.path.join(BASE_DIR, "local_data/NascentRNA")  
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "local_experiments")

# config files for creating data and running experiments
DATA_PARAMS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'scripts', 'data_params.py')
EXPERIMENT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'scripts', 'experiment_config.py')

CREDENTIALS_FILE = os.path.join(BASE_DIR, "tm_credentials")

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def convert_cell_cycle(arr, one_hot=False):
    if ('G1' in arr) or ('G2' in arr) or ('S' in arr):
        conv_arr = np.zeros(arr.shape, dtype=np.uint8)
        conv_arr[arr=='G1'] = 1
        conv_arr[arr=='S'] = 2
        conv_arr[arr=='G2'] = 3
        if one_hot:
            conv_arr = get_one_hot(conv_arr, 4)
    else:
        conv_arr = np.zeros(arr.shape, dtype=np.object)
        conv_arr[arr==1] = 'G1'
        conv_arr[arr==2] = 'S'
        conv_arr[arr==3] = 'G2'
    return conv_arr

def convert_perturbations(arr, one_hot=False):
    perturbations = ['AZD4573', 'CX5461', 'DMSO', 'Meayamycin', 'TSA', 'Triptolide', 'Wnt-C59',
 'normal']
    if np.isin(arr, perturbations).any():
        conv_arr = np.zeros(arr.shape, dtype=np.uint8)
        for i,p in enumerate(perturbations):
            conv_arr[arr==p] = i
        if one_hot:
            conv_arr = get_one_hot(conv_arr, 8)
    else:
        conv_arr = np.zeros(arr.shape, dtype=np.object)
        for i,p in enumerate(perturbations):
            conv_arr[arr==i]=p
    return conv_arr
