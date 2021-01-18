"""MPP_dataset_test2 dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
import sys

_DESCRIPTION = """
Dataset containing images of Multiplexed protein maps.
The elements of this data sets are multichannel images of singel cells alongside with its transcription rate.
This Dataset was builded after a preprocessing using the python script Transform_MPPData_into_images_no_split_from_script.ipynb. During this preprocessing the original MPPData was:
- Converted into images.
- Cleaned. Border and mitotic cells were removed.
- Normalized. Each channel was normalized using scale parameters obtained from the training set.
- Target value (scalar) calculated. The transcription rate was approximated taking the average of the measured pixels of the channel 00_EU. It is important to mention that the the target value was calculated BEFORE the normalization process.

This Dataset contains only the cells with no perturbations (i.e. cells such that perturbation in ['normal', 'DMSO']). Although perturbations 'TSA' seams not to have influence over the TR (00_EU avg), it seams to have an influence on the intensity of the channel 10_H3K27ac, and therefore it is not included. Only normal, DMSO (perturbations). Also, this TFDS keeps an equal number of cells in cell cycle G1, S, G2 across the train, val and test set by sampling with replacement.

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

class MPP_dataset_test2(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for MPP_dataset_test2 dataset."""
	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
		'1.0.0': 'First Release. Only normal, DMSO (perturbations). This TFDS keeps an equal number of cells in cell cycle G1, S, G2 across the train, val and test set by sampling with replacemnt.',
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

		# Get normalization parameters
		self.normalization_vals = self._get_normalization_values()

		# Save output of this script
		self._save_tfds_metadata()

		# To make return more readable
		img_size = self.pp_param['img_size']
		n_channels = self.input_ids.shape[0]

		return tfds.core.DatasetInfo(
			builder=self,
			description=_DESCRIPTION,
			features=tfds.features.FeaturesDict({
				# These are the features of your dataset like images, labels ...
				'mapobject_id_cell': tfds.features.Text(),
				'image': tfds.features.Tensor(shape=(img_size, img_size, n_channels), dtype=tf.float32),
				'target': tfds.features.Tensor(shape=(1,), dtype=tf.float64),
				}),

				supervised_keys=('image', 'target'),  # Set to `None` to disable
				homepage='https://www.helmholtz-muenchen.de/icb',
				citation=_CITATION,
			)

	def _split_generators(self, dl_manager: tfds.download.DownloadManager):
		"""
		downloads and splits data
		Returns SplitGenerators.
		This dataset was created following the guide:
		https://www.tensorflow.org/datasets/add_dataset
		"""

		input_data_path = dl_manager.extract(self.data_source_path)
		images_path = input_data_path / 'data'

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

		# Get file names
		file_names = [file.name for file in images_path.glob('*.npz')]
		# take imaes that are in the file system, the filtered_metadata df and
		# correspond to the desired subset (train, val, test)
		mask = (self.filtered_metadata.set == subset)
		filtered_cells = self.filtered_metadata.mapobject_id_cell[mask].values.astype(str)
		file_names = [fn for fn in file_names if fn[:-4] in filtered_cells]
		file_names = np.array(file_names)

		self.normalization_vals = self.normalization_vals.astype(np.float32)
		n_channels = self.input_ids.shape[0]
		img_size = self.pp_param['img_size']
		img_shape = (img_size, img_size, n_channels)

		for fn in file_names:
			cell = np.load(images_path.joinpath(fn))
			cell_id = int(fn.split('.')[0])
			cell_img = cell['img'][:,:,self.input_ids]
			cell_target = [cell['targets'][self.output_id]]
			# Normalize cell image
			cell_img = cell_img.astype(np.float32)
			cell_img = cell_img.reshape((-1, n_channels))
			cell_img /= self.normalization_vals
			cell_img = cell_img.reshape(img_shape)

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
		self.data_source_path = self.pp_param['output_data_dir']

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

		cell_cycles = self.tfds_param['cell_cycles']
		cc_samp_size = self.tfds_param['sample_size_per_cell_cycle']

		# same number of cells of each cell cycle will be added to each set
		n_train_cc = int(cc_samp_size * self.tfds_param['train_frac'])
		n_val_cc = int(cc_samp_size * self.tfds_param['val_frac'])
		n_test_cc = int(cc_samp_size - n_train_cc - n_val_cc)

		print('TFDS size: {}\n\ttrain: {}\n\tval: {}\n\ttest: {}'.format(len(cell_cycles)*(n_train_cc+n_val_cc+n_test_cc), len(cell_cycles)*n_train_cc, len(cell_cycles)*n_val_cc, len(cell_cycles)*n_test_cc))

		# create df containing the mapobject_id_cell and its set
		set_df = pd.DataFrame(columns=['mapobject_id_cell', 'set'])

		# create train, val and test partitions keeping the proportion of perturbation and cell_cycle
		for cc in cell_cycles:
			print('\nSampling {} cell cycle...'.format(cc))
			mask = (filtered_metadata.cell_cycle == cc)
			cell_ids = filtered_metadata[mask].mapobject_id_cell.values
			print('{} unique cells'.format(len(cell_ids)))

			np.random.seed(self.tfds_param['seed'])
			print('Sampling (with replacement) {} cells (from {}) for val set'.format(n_val_cc, len(cell_ids)))
			val_cell_ids = np.random.choice(cell_ids, n_val_cc, replace=True)
			remaining_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in val_cell_ids]

			print('Sampling (with replacement) {} cells (from {}) for test set'.format(n_test_cc, len(remaining_cell_ids)))
			test_cell_ids = np.random.choice(remaining_cell_ids, n_test_cc, replace=True)
			remaining_cell_ids = [cell_id for cell_id in remaining_cell_ids if cell_id not in test_cell_ids]

			print('Sampling (with replacement) {} cells (from {}) for train set'.format(n_train_cc, len(remaining_cell_ids)))
			train_cell_ids = np.random.choice(remaining_cell_ids, n_train_cc, replace=True)

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
		                        how='right')

		n_train = np.sum(filtered_metadata.set == 'train')
		n_val = np.sum(filtered_metadata.set == 'val')
		n_test = np.sum(filtered_metadata.set == 'test')
		print('Number of cells in train set: {}, val set: {}, test_set: {}\nTotal number of cells: {}\n'.format(n_train, n_val, n_test, n_train+n_val+n_test))

		return filtered_metadata

	def _get_normalization_values(self):
		"""
		Get the normalization rescale values for each selected input channel from the train partition and the given percentile.
	    Input:
	        input_channel_ids: list indicating for which channels we must calculate the normalization parameters.
	        percentile: integer in [0,100].
	    Output:
	        norm_vals: numpy array of lenght number_of_channels, containing the normalization values of each channel
		"""
		def apply_mask_to_channel(channel):
			return channel[cell_mask]

		cell_path = os.path.join(self.data_source_path, 'data')
		# Get file names
		file_names = os.listdir(cell_path)
		file_names = [name for name in file_names if name[-4:] == '.npz']

		# take imaes that are in the file system, the filtered_metadata df and
		# correspond to the train subset
		mask = (self.filtered_metadata.set == 'train')
		filtered_cells = self.filtered_metadata.mapobject_id_cell[mask].values.astype(str)
		file_names = [fn for fn in file_names if fn[:-4] in filtered_cells]
		file_names = np.array(file_names)

	    # The percentile per channel need to be taken accordingly to the mask. Since there is no way to concatenate np arrays in place (without creating a copy), we first create an array that will be filled with the values necessary to calculate the percentile per channel. Note that the size of this array is the same for every channel.
	    #https://stackoverflow.com/questions/7869095/concatenate-numpy-arrays-without-copying
		n_pixels = 0
		for fn in file_names:
			cell = np.load(os.path.join(cell_path, fn))
			n_pixels += cell['mask'].sum()
		print('Number of pixels for each channel and for all the training cells images:{}'.format(n_pixels))

		# Create the array that will hold the train dataset filtered by the mask
		n_channels = self.input_ids.shape[0]
		img_dtype = getattr(np, self.pp_param['images_dtype'])
		train_pixels = np.zeros((n_pixels, n_channels), dtype=img_dtype)
		print('Size of the array that holds the training pixels (GB): {}'.format(sys.getsizeof(train_pixels)/1e9))

		# Fill train_pixels with the train data
		idx_l = 0
		idx_h = 0
		for fn in file_names:
			cell = np.load(os.path.join(cell_path, fn))
			cell_mask = cell['mask'].reshape(-1)
			cell_img = cell['img'][:,:,self.input_ids]
			cell_img = cell_img.reshape((-1, n_channels))
			idx_l = idx_h
			idx_h += cell_mask.sum()

			train_pixels[idx_l:idx_h,:] = np.apply_along_axis(apply_mask_to_channel, 0, cell_img)

		norm_vals = np.percentile(train_pixels, self.tfds_param['percentile'], axis=0)

		return norm_vals


	def _get_IO_channel_ids(self):

		# Input ids
		input_channels = self.tfds_param['input_channels']
		input_ids = self.channels.set_index('name').loc[input_channels]['channel_id'].values

		# Output id
		output_channel = self.tfds_param['output_channels']
		output_id = self.channels.set_index('name').loc[output_channel]['channel_id'].values[0]

		return input_ids, output_id

	def _save_tfds_metadata(self):

		# Get the path where the script is executed
		script_dir = os.path.realpath(__file__)
		script_dir = os.path.dirname(script_dir)
		# Path to save output of this script (params, full_metadata, etc. NO tfds!)
		script_output_path = os.path.join(script_dir, 'Output')

		# Add info to the description
		global _DESCRIPTION
		_DESCRIPTION += '\ninput_channels:\n{}'.format(self.tfds_param['input_channels'])
		_DESCRIPTION += '\n\noutput_channel:\n{}'.format(self.tfds_param['output_channels'])
		_DESCRIPTION += '\n\nNormalization values:\n{}'.format(self.normalization_vals)
		_DESCRIPTION += '\n\nMore information about this TFDS (metadata, metadata after filtering, dataset creation parameters and channel info including normalization values) can be found on:\n{}\n\n'.format(script_output_path)

		# save original metadata:
		with open(os.path.join(script_output_path, 'metadata.csv'), 'w') as file:
			self.full_metadata.to_csv(file, index=False, sep=',')

		# Save filtered metadata
		with open(os.path.join(script_output_path, 'filtered_metadata.csv'), 'w') as file:
			self.filtered_metadata.to_csv(file, index=False, sep=',')

		# save tfds creation parameters
		with open(os.path.join(script_output_path, 'tfds_parameters.json'), 'w') as file:
			json.dump(self.tfds_param, file, indent=4)

		# save data preprocessing parameters
		with open(os.path.join(script_output_path, 'data_pp_parameters.json'), 'w') as file:
			json.dump(self.pp_param, file, indent=4)

		# Save channel file:
		# First add the normalization values to the output channel file
		norm_vals_df = pd.DataFrame(self.input_ids, columns=['channel_id'])
		norm_vals_df['normalization_vals'] = self.normalization_vals
		norm_vals_df['type'] = 'input'

		mask = self.channels.channel_id.isin(self.input_ids)
		self.channels = self.channels[mask]
		self.channels = self.channels.merge(norm_vals_df,
											left_on='channel_id',
											right_on='channel_id',
											how='left')
		temp_dict = {'channel_id':self.output_id, 'name':(self.tfds_param['output_channels'])[0], 'normalization_vals':1, 'type':'output'}
		self.channels = self.channels.append(temp_dict, ignore_index=True)

		with open(os.path.join(script_output_path, 'channels.csv'), 'w') as file:
			self.channels.to_csv(file, index=False, sep=',')
