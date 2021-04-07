import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys
import json
import copy
import time
import argparse
import logging
import socket

# Load external libraries
key = 'external_libs_path'
if socket.gethostname() == 'hughes-machine':
    external_libs_path = '/home/hhughes/Documents/Master_Thesis/Project/workspace/libs'
else:
    external_libs_path= '/storage/groups/ml01/code/andres.becker/master_thesis/workspace/libs'
print('External libs path: \n'+external_libs_path)

if not os.path.exists(external_libs_path):
    msg = 'External library path {} does not exist!'.format(external_libs_path)
    raise Exception(msg)

# Add EXTERNAL_LIBS_PATH to sys paths (for loading libraries)
sys.path.insert(1, external_libs_path)
from Utils import print_stdout_and_log as printc
from Utils import set_GPU_config as set_GPU_config
import NN_interpretability as nn_inter
import Data_augmentation
import tfds_utils

parser_grn = argparse.ArgumentParser(description='Compute the score map (VarGrad IG) of a cell or group of cells')

parser_grn.add_argument('-i', '--input_parameters_file',
        required=True,
        type=str,
        help='Absolute path to the input pararameters file.')

args = parser_grn.parse_args()
PARAMETERS_FILE = args.input_parameters_file

if not os.path.exists(PARAMETERS_FILE):
    raise Exception('Parameter file {} does not exist!'.format(PARAMETERS_FILE))
else:
    print('Input parameters file: ', PARAMETERS_FILE)

# Load parameters
with open(PARAMETERS_FILE, 'r') as file:
    p = json.load(file)
if 'save_df_only' not in p.keys():
    p['save_df_only'] = False
print(p.keys())

# Set logging
log_file_path = p['log_file_name']
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    level=getattr(logging, 'INFO')
)
msg = 'Parameters loaded from file:\n{}'.format(PARAMETERS_FILE)
printc(msg)

set_GPU_config(p['disable_gpu'], p['set_memory_growth'])

# Load model parameters
temp_path = os.path.join(p['model_path'], 'parameters.json')
with open(temp_path, 'r') as file:
    model_p = json.load(file)
printc('Model Parameters')
printc(model_p.keys())

# Set script vars
output_path = os.path.join(p['output_data_dir'], p['output_dir_name'])
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
print('\nOutput path:\n{}'.format(output_path))

# Load TFDS
dataset, ds_info = tfds.load(
    name=model_p['tf_ds_name'],
    data_dir=model_p['local_tf_datasets'],
    # If False, returns a dictionary with all the features
    as_supervised=False,
    shuffle_files=False,
    with_info=True)
# Load TFDS metadata
tfds_metadata = tfds_utils.Costum_TFDS_metadata().load_metadata(ds_info.data_dir)
tfds_metadata.keys()
msg = 'Tensorflow dataset {} loaded from:\n{}'.format(model_p['tf_ds_name'], model_p['local_tf_datasets'])
printc(msg)

# Load splits
train_data, val_data, test_data = dataset['train'], dataset['validation'], dataset['test']

# check if cell ids given, if not, add all ids from the selected partitons
msg = 'Loading cell ids'
if 'cell_ids_file' in p.keys():
    with open(p['cell_ids_file'], 'r') as file:
        temp_df = pd.read_csv(file, names=['cell_ids'])
        cell_ids = temp_df.cell_ids.astype(str).values
else:
    mask = tfds_metadata['metadata_df'].set.isin(p['ds_partitions'])
    cell_ids = tfds_metadata['metadata_df']['mapobject_id_cell'][mask].astype(str).values
msg += '\n\t mapobject_id_cell to be processed:\n\t {}\n\n'.format(cell_ids)
printc(msg)

# Get input_ids
selected_channels = model_p['input_channels']
input_ids = np.array(tfds_metadata['channels_df'].set_index(['name']).loc[selected_channels].TFDS_channel_id.values)
input_ids = input_ids.astype(np.int16)
msg = 'Corresponding input channel ids:\n{}'.format(input_ids)
printc(msg)

# DF to store the stddev per channel (sum of pixels) and for the whole cell
channel_names = list(tfds_metadata['channels_df'].set_index('TFDS_channel_id').iloc[input_ids].name.values)
col_name = channel_names + ['Total_cell_stddev']
score_map_stddev_df = pd.DataFrame(columns=col_name+['Model', 'mapobject_id_cell'])

# Load model
temp_path = os.path.join(p['model_path'], 'model',p['model_CMA'])
model = tf.keras.models.load_model(temp_path)
msg = '\nModel loaded successfully:\n'
printc(msg)
model.summary(print_fn=printc)

# Load cell, compute its Score map and save it
n_cells = len(cell_ids)
dss = [train_data, val_data, test_data]
ds_names = ['train', 'val', 'test']
cell_count = 0
for ds, dsn in zip(dss, ds_names):
    for cell in ds:
        # get cell id
        cell_id = cell['mapobject_id_cell'].numpy()
        cell_id = str(cell_id.decode())

        if cell_id in cell_ids:
            cell_count += 1
            printc('Processing cell: {}, {}/{}'.format(cell_id, cell_count, n_cells))

            # get cell img
            cell_img = cell['image'].numpy()

            # Get cell mask
            n_channels = cell_img.shape[-1]
            cell_mask = cell_img[:,:,-1]
            cell_mask = np.repeat(cell_mask[:,:,None], n_channels, axis=2)
            cell_mask = cell_mask.astype(np.bool)

            printc('\t Computing Cell Score map...')
            tic = time.time()
            temp_score_map = nn_inter.get_VarGrad(img=cell_img,
                                                  img_mask=cell_mask,
                                                  baseline=p['IG_baseline'],
                                                  IG_m_steps=p['IG_m_steps'],
                                                  model=model,
                                                  n_images=p['VarGrad_n_samples']
                                                 )
            temp_score_map = temp_score_map.numpy()
            # Filter Score map channels
            temp_score_map = temp_score_map[:,:,input_ids]
            # Compute sum of stddevs per channel and save it
            score_channel_std = []
            total_var = 0
            for c in input_ids:
                score_channel_std.append(np.sum(temp_score_map[:,:,c]))
            total_var = np.sum(score_channel_std)
            # regularize values before saving
            score_channel_std = np.array(score_channel_std) / total_var
            temp_data = np.concatenate((score_channel_std, [total_var]), axis=0).reshape((1,-1))
            temp_df = pd.DataFrame(temp_data, columns=col_name)
            temp_df['Model'] = p['output_dir_name']
            temp_df['mapobject_id_cell'] = cell_id
            score_map_stddev_df = pd.concat([score_map_stddev_df, temp_df], ignore_index=True)

            tac = time.time()
            if not p['save_df_only']:
                msg = '\t Score map computed in {} seconds'.format(round(tac-tic,2))
                printc(msg)
                msg = '\t Saving score map...'
                printc(msg)
                file_path = os.path.join(output_path, 'data', p['output_name_prefix']+cell_id+'.npy')
                np.save(file_path, temp_score_map)

# Save DF with comulative stddevs
temp_path = os.path.join(output_path, 'score_map_stddev.csv')
with open(temp_path, 'w') as file:
    score_map_stddev_df.to_csv(file, index=False)
