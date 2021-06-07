"""MPP_DS_normal_DMSO_z_score dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
import sys
import socket

# Load external libraries
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
import tfds_utils

_DESCRIPTION = """
Dataset containing images of Multiplexed protein maps.
The elements of this data sets are multichannel images of singel cells alongside with its transcription rate. The cell mask is saved on the last channel of the cell image.
This Dataset was builded after a preprocessing using the python script MPPData_into_images_no_split.ipynb. During this preprocessing the original MPPData was:
- Converted into image, mask and target vector.
- Cleaned. Border and mitotic cells were removed.

This Dataset contains only the cells with no perturbations (i.e. cells such that perturbation in ['normal', 'DMSO']). Although perturbations 'TSA' seams not to have influence over the TR (00_EU avg), it seams to have an influence on the intensity of the channel H3K27ac, and therefore it is not included. Only normal, DMSO (perturbations).

"""

_CITATION = """
@article {Guteaar7042,
author = {Gut, Gabriele and Herrmann, Markus D. and Pelkmans, Lucas},
	title = {Multiplexed protein maps link subcellular organization to cellular states},
	volume = {361},
	number = {6401},
	elocation-id = {eaar7042},
	year = {2018},
	doi = {10.1126/science.aar7042},
	publisher = {American Association for the Advancement of Science},
	issn = {0036-8075},
	URL = {https://science.sciencemag.org/content/361/6401/eaar7042},
	eprint = {https://science.sciencemag.org/content/361/6401/eaar7042.full.pdf},
	journal = {Science}
}
"""

class MPP_DS_normal_DMSO_z_score(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for MPP_DS_normal_DMSO_z_score dataset."""
	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
		'1.0.0': 'First Release. Only normal, DMSO (perturbations).',
		}

	def _info(self) -> tfds.core.DatasetInfo:
		"""Returns the dataset metadata."""

		# Load necessary files to create the TFDS
		self._load_files()

		# Get filtered metadata (acordinglly with the given perturbations and cell cycle)
		filtered_metadata = self._get_filtered_metadata()

		# add train, val and test partitions info to filtered_metadata
		self.filtered_metadata = self._add_train_val_test_split_to_metadata(filtered_metadata)

		# Get channel ids corresponding to the input and the output
		self.input_ids, self.output_id = self._get_IO_channel_ids()

		# Get the train per-channel parameters (percentile, mean, stddev)
		self.train_params = self._get_train_per_channel_parameters(self.tfds_param['percentile'])

		# Save extra info into the TDFS metadata (channels_df, metadata_df, input params, etc.)
		metadata, meta_description = self._create_metadata()

		# To make return more readable
		img_size = self.pp_param['img_size']
		# since cell mask will be saved in the last channel, we add 1
		n_channels = self.input_ids.shape[0] + 1

		return tfds.core.DatasetInfo(
			builder = self,
			description = _DESCRIPTION + meta_description,
			features = tfds.features.FeaturesDict({
				# These are the features of your dataset like images, labels ...
				'mapobject_id_cell': tfds.features.Text(),
				'image': tfds.features.Tensor(shape=(img_size, img_size, n_channels), dtype=tf.float32),
				'target': tfds.features.Tensor(shape=(1,), dtype=tf.float64),
				}),

			supervised_keys = ('image', 'target'),  # Set to `None` to disable
			homepage = 'https://www.helmholtz-muenchen.de/icb',
			citation = _CITATION,
			metadata = metadata,
			)

	def _split_generators(self, dl_manager: tfds.download.DownloadManager):
		"""
		downloads and splits data
		Returns SplitGenerators.
		This dataset was created following the guide:
		https://www.tensorflow.org/datasets/add_dataset
		"""

		input_data_path = dl_manager.extract(self.data_source_path)
		images_path = input_data_path / self.pp_param['output_pp_data_dir_name']

		return {
			'train': self._generate_examples(images_path, 'train'),
			'validation': self._generate_examples(images_path, 'val'),
        	'test': self._generate_examples(images_path, 'test'),
				}

	def _generate_examples(self, images_path, subset):
		"""
	  	What is yield?
	  	https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	  	generates the examples for each split from the source data. Yields examples.
	  	"""

		# Get cell ids in metadata, subset and that are available in the FS
		subset_ids = self._get_partitions_ids(partitions=[subset], data_dir=str(images_path.resolve()))

		# Get file names
		file_names = [cell_id+'.npz' for cell_id in subset_ids]
		file_names = np.array(file_names)

		for fn in file_names:
			cell = np.load(images_path.joinpath(fn))
			cell_id = int(fn.split('.')[0])
			cell_img = cell['img'][:,:,self.input_ids]
            cell_img = cell_img.astype(np.float32)
			cell_mask = cell['mask']
			cell_target = [cell['targets'][self.output_id]]

			# Apply preprocessing to each cell image before saving
			cell_img = self._apply_preprocessing(cell_img, cell_mask)

			# Save mask in the last channel of the image, this is needed for some data augmentation techniques.
			cell_mask = np.expand_dims(cell_mask, axis=-1)
			cell_mask = cell_mask.astype(np.float32)
			# add mask as last channel of the cell image
			cell_img = np.concatenate((cell_img, cell_mask), axis=-1)

			yield cell_id, {
				'mapobject_id_cell': str(cell_id),
				'image': cell_img,
				'target': cell_target,
				}

	def _load_files(self):
		# Load tf dataset parameters
		with open(os.path.join('./Parameters', 'tf_dataset_parameters.json')) as file:
			self.tfds_param = json.load(file)

		# Load parameters used to generate preprocessed data
		with open(self.tfds_param['data_source_parameters'], 'r') as file:
			self.pp_param = json.load(file)

		# Path where the preprocessed data to be transformed into tf dataset is
		self.data_source_path = self.pp_param['output_pp_data_path']

		# Load metadata file:
		with open(os.path.join(self.data_source_path, 'metadata.csv'), 'r') as file:
			self.full_metadata =  pd.read_csv(file)

		# Load channel file:
		with open(os.path.join(self.data_source_path, 'channels.csv')) as file:
			self.channels =  pd.read_csv(file)

	def _get_filtered_metadata(self):

		# leave only unique elements
		self.full_metadata = self.full_metadata.groupby(['mapobject_id_cell'], sort=False, as_index=False).first()

		# get mask acordinglly to the selected perturbations and cell cycle
		mask = self.full_metadata.perturbation.isin(self.tfds_param['perturbations'])
		mask &= self.full_metadata.cell_cycle.isin(self.tfds_param['cell_cycles'])
		# Filter metadata
		filtered_metadata = self.full_metadata[mask].copy()

		return filtered_metadata

	def _add_train_val_test_split_to_metadata(self, filtered_metadata):

		# create df containing the mapobject_id_cell and its set
		set_df = pd.DataFrame(columns=['mapobject_id_cell', 'set'])

		cell_ids = filtered_metadata.mapobject_id_cell.unique()

        # for testing
		#cell_ids = cell_ids[0:10]

		# Create split sizes (same for all the cell cycles)
		n_train = int(len(cell_ids) * self.tfds_param['train_frac'])
		n_val = int(len(cell_ids) * self.tfds_param['val_frac'])
		n_test = len(cell_ids) - n_train - n_val

		# get the mapobject_id_cell belonging to the train, val and test partitions
		np.random.seed(self.tfds_param['seed'])
		np.random.shuffle(cell_ids)
		train_cell_ids = cell_ids[0:n_train]
		val_cell_ids = cell_ids[n_train:n_train+n_val]
		test_cell_ids = cell_ids[n_train+n_val:n_train+n_val+n_test]

		# Save cell id with its corresponding set
		temp_df = pd.DataFrame(train_cell_ids, columns=['mapobject_id_cell'])
		temp_df['set'] = 'train'
		set_df = pd.concat((set_df, temp_df), ignore_index=True)
		temp_df = pd.DataFrame(val_cell_ids, columns=['mapobject_id_cell'])
		temp_df['set'] = 'val'
		set_df = pd.concat((set_df, temp_df), ignore_index=True)
		temp_df = pd.DataFrame(test_cell_ids, columns=['mapobject_id_cell'])
		temp_df['set'] = 'test'
		set_df = pd.concat((set_df, temp_df), ignore_index=True)

		# merge the filtered_metadata with the df containing the set
		filtered_metadata = filtered_metadata.merge(set_df,
		                        left_on='mapobject_id_cell',
		                        right_on='mapobject_id_cell',
		                        how='left')

		n_train = np.sum(filtered_metadata.set == 'train')
		n_val = np.sum(filtered_metadata.set == 'val')
		n_test = np.sum(filtered_metadata.set == 'test')
		print('Number of cells in train set: {}, val set: {}, test_set: {}\nTotal number of cells: {}\n'.format(n_train, n_val, n_test, n_train+n_val+n_test))

		return filtered_metadata

	def _apply_clipping(self, img, clipping_values):
		"""
		Apply cliping to all partitions using the percentile_vec
		"""
		for c in range(clipping_values.shape[0]):
			temp_mask = (img[:,:,c] > clipping_values[c])
			img[:,:,c][temp_mask] = clipping_values[c]

		return img

	def _apply_z_score(self, img, cell_mask, mean_vals, stddev_vals):
		"""
		Apply z-scoring to all partitions using the masked per-channel train mean and stddev.
		"""
		for c in range(mean_vals.shape[0]):

			img[:,:,c][cell_mask] -= mean_vals[c]
			img[:,:,c][cell_mask] /= stddev_vals[c]

		return img

	def _apply_preprocessing(self, img, cell_mask):
		"""
		Apply data preprocessing specified in the tfds parameters file
		"""

		# Apply clipping if selected
		if self.tfds_param['apply_clipping']:
			img = self._apply_clipping(img, self.train_params['percentile'])

		# Apply rescaling if selected
		if self.tfds_param['apply_linear_scaling']:
			img /= self.train_params['percentile']

		# Apply mean extraction if selected
		if self.tfds_param['apply_mean_extraction']:
			img[cell_mask] -= self.train_params['mean']

		# Apply z_score
		if self.tfds_param['apply_z_score']:
			img = self._apply_z_score(img, cell_mask, self.train_params['mean'], self.train_params['stddev'])

		return img

	def _get_partitions_ids(self, partitions=['train'], data_dir=None, file_type='npz'):
		"""
		Get train ids given the metadata and the available files on disk
		"""

		# Get ids on the metadata and in the given partitions
		mask = self.filtered_metadata.set.isin(partitions)
		metadata_ids = set(self.filtered_metadata.mapobject_id_cell[mask].values.astype(str))

		# Get ids of available cell on the file system
		file_ids = set([id.split(".")[0] for id in os.listdir(data_dir) if id.split(".")[1] == file_type])

		# Oly return ids that are in the metadata (and belong to given partitions) and are available in the FS
		return list(file_ids.intersection(metadata_ids))

	def _load_imgs_pixels(self, train_ids):
		"""
		Return a np array containing the masked pixels for all the cells in train_ids and channels in self.input_ids
		"""

		# get image data type
		dtype = getattr(np, self.pp_param['images_dtype'])

		# get total number of pixels to allocate np array
		# The percentile per channel need to be taken accordingly to the mask. Since there is no way to concatenate np arrays in place (without creating a copy), we first create an array that will be filled with the values necessary to calculate the percentile per channel. Note that the size of this array is the same for every channel.
	    #https://stackoverflow.com/questions/7869095/concatenate-numpy-arrays-without-copying
		data_dir = os.path.join(self.data_source_path, self.pp_param['output_pp_data_dir_name'])
		n_pixels_per_channel = 0
		for cell_id in train_ids:
			temp_path = os.path.join(data_dir, cell_id+'.npz')
			temp_cell = np.load(temp_path)
			n_pixels_per_channel += temp_cell['mask'].sum()
		print('Number of pixels for each channel and for all the training cells images:{}'.format(n_pixels_per_channel))

		# load pixel from all cells in train_ids and channels in self.input_ids
		imgs_pixels = np.zeros((n_pixels_per_channel, len(self.input_ids))).astype(dtype)
		print('Size of the array that holds the training pixels (GB): {}'.format(sys.getsizeof(imgs_pixels)/1e9))
		low_idx = 0
		up_idx = 0
		for i, cell_id in enumerate(train_ids):

			if (i % 100) == 0:
				print('Loading cell {}/{}'.format(i+1, len(train_ids)))

			temp_path = os.path.join(data_dir, cell_id+'.npz')
			temp_cell = np.load(temp_path)
			temp_img = temp_cell['img'][:,:,self.input_ids].astype(dtype)
			temp_mask = temp_cell['mask']

			low_idx = up_idx
			up_idx += temp_mask.sum()
			imgs_pixels[low_idx:up_idx, :] = temp_img[temp_mask]

		return imgs_pixels


	def _get_train_per_channel_parameters(self, percentile):
		"""
		Get the train per-channnel parameters.
		"""
		train_params = {}

		# Get cell ids in metadata, train set and that are available in the FS
		data_dir = os.path.join(self.data_source_path, self.pp_param['output_pp_data_dir_name'])
		train_ids = self._get_partitions_ids(partitions=['train'], data_dir=data_dir)

		# Get all train pixels accordingly to train_ids, input_ids (input channels) and cell masks
		train_pixels = self._load_imgs_pixels(train_ids)

		train_params['percentile'] = np.percentile(train_pixels, percentile, axis=0).astype(np.float32)

        # if clipping was selected, apply it to the train_pixels before getting the parameters
		if self.tfds_param['apply_clipping']:
			for c in range(train_params['percentile'].shape[0]):
				temp_mask = (train_pixels[:,c] > train_params['percentile'][c])
				train_pixels[:,c][temp_mask] = train_params['percentile'][c]

		# if lin scaling was selected, then apply it before
		if self.tfds_param['apply_linear_scaling']:
			train_pixels /= train_params['percentile']

		train_params['mean'] = train_pixels.mean(axis=0)
		train_params['stddev'] = train_pixels.std(axis=0)

		return train_params

	def _get_IO_channel_ids(self):

		# Input ids
		input_channels = self.tfds_param['input_channels']
		input_ids = self.channels.set_index('name').loc[input_channels]['channel_id'].values

		# Output id
		output_channel = self.tfds_param['output_channels']
		output_id = self.channels.set_index('name').loc[output_channel]['channel_id'].values[0]

		return input_ids, output_id

	def _create_metadata(self):
		"""
		This function create an instances of the class tfds.core.Metadata ment to save extra information into the created TFDS dir. To load the info saved here, load library tfds_utils.py lib and instantiate the class Costum_TFDS_metadata and execut the method load_metadata(TFDS_data_dir).
        """
		info = '\nTo load more info about this TDFS (cell metadata df, channels df containing preprocessing parameters, the used TFDS and preprocessing arameters) after loading this TFDS run:'
		info += '\nimport tfds_utils'
		info += '\n# Load TFDS and TFDS info'
		info += '\ndataset, ds_info = tfds.load(...'
		info += '\n\nmetadata = tfds_utils.Add_extra_info_to_TFDS()'
		info += '\nmetadata.load_metadata(ds_info.data_dir)'
		info += '\n-'
		info += '\n-'

		# create channel file containing channels name id and preprocessing parameters
		# First add the per-channel percentile values and per channel train mean to the output channel file
		temp_df = pd.DataFrame(self.input_ids, columns=['channel_id'])
		percentil_col_name = 'train_'+str(self.tfds_param['percentile'])+'_percentile'
		temp_df[percentil_col_name] = self.train_params['percentile']
		mean_col_name = 'train_mean_after_clipping'
		temp_df[mean_col_name] = self.train_params['mean']
		stddev_col_name = 'train_stddev_after_clipping'
		temp_df[stddev_col_name] = self.train_params['stddev']
		temp_df['type'] = 'input'

		mask = self.channels.channel_id.isin(self.input_ids)
		self.channels = self.channels[mask]
		self.channels = self.channels.merge(temp_df,
											left_on='channel_id',
											right_on='channel_id',
											how='left')
		self.channels = self.channels.rename(columns={'channel_id': 'original_channel_id'})
		self.channels['TFDS_channel_id'] = range(len(self.input_ids))

		# save info about the cell mask
		temp_dict = {'original_channel_id': 'NaN', 'TFDS_channel_id': len(self.input_ids), 'name':'cell_mask', percentil_col_name:'NaN', mean_col_name:'NaN', stddev_col_name:'NaN', 'type':'NaN'}
		self.channels = self.channels.append(temp_dict, ignore_index=True)

		# save info about output channel
		temp_dict = {'original_channel_id': self.output_id, 'TFDS_channel_id': 'NaN', 'name':(self.tfds_param['output_channels'])[0], percentil_col_name:'NaN', mean_col_name:'NaN', stddev_col_name:'NaN', 'type':'output'}
		self.channels = self.channels.append(temp_dict, ignore_index=True)

		# intance of tfds.core.Metadata to store more info about this TFDS
        # If you want to save more info, finish the dict key with 'parameters' to save it as json or with 'df' to save it as csv.
		temp_dict = {}
		temp_dict['tfds_creation_parameters'] = self.tfds_param
		temp_dict['data_pp_parameters'] = self.pp_param
		temp_dict['metadata_df'] = self.filtered_metadata
		temp_dict['channels_df'] = self.channels

		return tfds_utils.Costum_TFDS_metadata(temp_dict), info
