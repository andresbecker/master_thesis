import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import json
import copy
import time
import argparse
import logging

def print_stdout_and_log(msg):
    logging.info(msg)
    print(msg)


parser_grn = argparse.ArgumentParser(description='Compute the score map (VarGrad IG) of a cell or group of cells')

parser_grn.add_argument('-i', '--input_parameters_file',
        required=True,
        type=str,
        help='Absolute path to the input pararameters file.')

parser_grn.add_argument('--cell_id',
        required=False,
        default=None,
        type=str,
        help='If cell_id is given, then only that cell is processed.')

args = parser_grn.parse_args()
PARAMETERS_FILE = args.input_parameters_file

if not os.path.exists(PARAMETERS_FILE):
    raise Exception('Parameter file {} does not exist!'.format(PARAMETERS_FILE))
else:
    print('Input parameters file: ', PARAMETERS_FILE)

# Load parameters
with open(PARAMETERS_FILE, 'r') as file:
    p = json.load(file)
print(p.keys())

# Set logging
log_file_path = p['log_file_name']
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    level=getattr(logging, 'INFO')
)
msg = 'Parameters loaded from file:\n{}'.format(PARAMETERS_FILE)
print_stdout_and_log(msg)

# Load external libraries path
EXTERNAL_LIBS_PATH = p['external_libs_path']
if not os.path.exists(EXTERNAL_LIBS_PATH):
    msg = 'External library path {} does not exist!'.format(EXTERNAL_LIBS_PATH)
    logging.error(msg)
    raise Exception(msg)
else:
    msg='EXTERNAL_LIBS_PATH: {}'.format(EXTERNAL_LIBS_PATH)
    print_stdout_and_log(msg)
# Add EXTERNAL_LIBS_PATH to sys paths (for loading libraries)
sys.path.insert(1, EXTERNAL_LIBS_PATH)
# Load cortum libs
import NN_interpretability as nn_inter
import Data_augmentation as data_aug

# Make tf to ignore GPU
if p['disable_gpu']:
    msg = "Cuda devices (GPUs) disabled"
    logging.info(msg)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
msg = 'Physical GPU devises:\n{}'.format(physical_devices)
print_stdout_and_log(msg)

# Set script vars
output_path = os.path.join(p['output_data_dir'], p['output_dir_name'])
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
print('\nOutput path:\n{}'.format(output_path))

# Load Channels
with open(p['channels_file'], 'r') as file:
    channels_df = pd.read_csv(file)
# Get input channel ids
mask = channels_df.name.isin(p['input_channels'])
input_ids = channels_df[mask].channel_id.values
# Get normalization values
norm_vals = channels_df.sort_values(by=['channel_id']).normalization_vals.values
msg = 'Channel file {} loaded'.format(p['channels_file'])
print_stdout_and_log(msg)

# Load model
model = tf.keras.models.load_model(p['model_path'])
msg = '\nModel loaded successfully:\n'
print_stdout_and_log(msg)
model.summary(print_fn=print_stdout_and_log)

# check if cell ids given individually or in a file
if args.cell_id is not None:
    msg = 'Cell id {} given as argument, ignoring file {}'.format(args.cell_id, p['cell_ids_file'])
    cell_ids = [args.cell_id]
else:
    msg = 'Reading cell ids from file {}'.format(p['cell_ids_file'])
    with open(p['cell_ids_file'], 'r') as file:
        temp_df = pd.read_csv(file, names=['cell_ids'])
        cell_ids = temp_df.cell_ids.astype(str).values
msg += '\n\t mapobject_id_cell to be processed:\n\t {}\n\n'.format(cell_ids)
print_stdout_and_log(msg)

# DF to store the stddev per channel (sum of pixels) and for the whole cell
channel_names = list(channels_df.set_index('channel_id').iloc[input_ids].name.values)
col_name = channel_names + ['Total_cell_stddev']
score_map_stddev_df = pd.DataFrame(columns=col_name+['Model', 'mapobject_id_cell'])

# Load cell, compute its Score map and save it
n_cells = len(cell_ids)
for i, cell in enumerate(cell_ids, 1):
    print_stdout_and_log('Processing cell: {}, {}/{}'.format(cell, i, n_cells))
    print_stdout_and_log('\t Loading cell image...')
    temp_path = os.path.join(p['input_data_dir'], cell+'.npz')
    temp_cell = np.load(temp_path)
    # Normalize cell
    temp_img = copy.deepcopy(temp_cell['img'] / norm_vals)
    # filter accordingly to the input channels
    temp_img = temp_img[:,:,input_ids].astype(np.float32)
    if p['use_cell_mask']:
        print_stdout_and_log('\t Loading cell mask...')
        temp_mask = copy.deepcopy(temp_cell['mask'])
        # make cell mask to matchs cell image shape
        temp_mask = np.repeat(temp_mask[:,:,None], temp_img.shape[-1], axis=2)
    else:
        temp_mask = None
        p['output_name_prefix'] = 'no_mask_' + p['output_name_prefix']

    print_stdout_and_log('\t Computing Cell Score map...')
    tic = time.time()
    temp_score_map = nn_inter.get_VarGrad(img=temp_img,
                                          img_mask=temp_mask,
                                          baseline=p['IG_baseline'],
                                          IG_m_steps=p['IG_m_steps'],
                                          model=model,
                                          n_images=p['VarGrad_n_samples']
                                         )

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
    temp_df['mapobject_id_cell'] = cell
    score_map_stddev_df = pd.concat([score_map_stddev_df, temp_df], ignore_index=True)

    tac = time.time()
    msg = '\t Score map computed in {} seconds'.format(round(tac-tic,2))
    print_stdout_and_log(msg)
    msg = '\t Saving score map...'
    print_stdout_and_log(msg)
    file_path = os.path.join(output_path, 'data', p['output_name_prefix']+cell+'.npy')
    np.save(file_path, temp_score_map.numpy())

# Save DF with comulative stddevs
temp_path = os.path.join(output_path, 'score_map_stddev.csv')
with open(temp_path, 'w') as file:
    score_map_stddev_df.to_csv(file, index=False)
