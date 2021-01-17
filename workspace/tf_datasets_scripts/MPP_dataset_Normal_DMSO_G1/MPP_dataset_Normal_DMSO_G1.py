"""MPP_dataset_Normal_DMSO_G1 dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json

_DESCRIPTION = """
Dataset containing images of Multiplexed protein maps.
The elements of this data sets are multichannel images of singel cells alongside with its transcription rate.
This Dataset was builded after a preprocessing using the python script Transform_MPPData_into_images_from_script.ipynb. During this preprocessing the original MPPData was:
- Converted into images.
- Cleaned. Border and mitotic cells were removed.
- Normalized. Each channel was normalized using scale parameters obtained from the training set.
- Target value (scalar) calculated. The transcription rate was approximated taking the average of the measured pixels of the channel 00_EU. It is important to mention that the the target value was calculated BEFORE the normalization process.

This Dataset contains only the cells with no perturbations (i.e. cells such that perturbation in ['normal', 'DMSO']). Although perturbations 'TSA' seams not to have influence over the TR (00_EU avg), it seams to have an influence on the intensity of the channel 10_H3K27ac, and therefore it is not included. Besides, this dataset only contain cell in G1 state.

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

class MPP_dataset_Normal_DMSO_G1(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for MPP_dataset_Normal_DMSO_G1 dataset."""
	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
		'1.0.0': 'First Release. Only normal, DMSO (perturbations) and Cells in G1 state.',
		}

	def _info(self) -> tfds.core.DatasetInfo:
		"""Returns the dataset metadata."""

		# Get the path where the script is executed
		script_dir = os.path.realpath(__file__)
		script_dir = os.path.dirname(script_dir)

		# Path to save output of this script (params, full_metadata, etc. NO tfds!)
		self.script_output_path = os.path.join(script_dir, 'Output')

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
			channels =  pd.read_csv(file)


		# images data type
		img_dtype = getattr(tf, self.pp_param['images_dtype'])

		# Get info from the preprocessing parameters
		img_size = self.pp_param['img_size']

		# Which perturbations to use?
		perturbations = self.tfds_param['perturbations']

		# Which cell cycles to use?
		cell_cycles = self.tfds_param['cell_cycles']

		# leave only unique elements
		self.full_metadata = self.full_metadata.groupby(['mapobject_id_cell'], sort=False, as_index=False).first()
		# filter acordinglly to the selected perturbations
		self.filtered_metadata = self.full_metadata[self.full_metadata.perturbation.isin(perturbations) & self.full_metadata.cell_cycle.isin(cell_cycles)]

		# create df containing the mapobject_id_cell and its set
		set_df = pd.DataFrame(columns=['mapobject_id_cell', 'set'])

		# create train, val and test partitions keeping the proportion of perturbation and cell_cycle
		for per in perturbations:
		    for cc in cell_cycles:
		        # Create mask that contains the cells with the corresponding perturbation and cell_cycle
		        mask = (self.filtered_metadata.perturbation == per) & (self.filtered_metadata.cell_cycle == cc)
		        # get mapobject_id_cell and the size of the train, val and test partitions
		        cell_ids = self.filtered_metadata[mask].mapobject_id_cell.values
		        n_train = int(len(cell_ids) * self.tfds_param['train_frac'])
		        n_val = int(len(cell_ids) * self.tfds_param['val_frac'])
		        n_test = len(cell_ids) - n_train - n_val
		        print('For partition (perturbation, cell_cycle)=({}, {}):'.format(per, cc))
		        print('\tNumber of cells in train set: {}, val set: {}, test_set: {}\n\tTotal number of cells: {}\n'.format(n_train, n_val, n_test, n_train+n_val+n_test))

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
		self.filtered_metadata = self.filtered_metadata.merge(set_df,
		                        left_on='mapobject_id_cell',
		                        right_on='mapobject_id_cell',
		                        how='left')

		n_train = np.sum(self.filtered_metadata.set == 'train')
		n_val = np.sum(self.filtered_metadata.set == 'val')
		n_test = np.sum(self.filtered_metadata.set == 'test')
		print('Number of cells in train set: {}, val set: {}, test_set: {}\nTotal number of cells: {}\n'.format(n_train, n_val, n_test, n_train+n_val+n_test))

		input_channels = self.tfds_param['input_channels']
		self.input_ids = channels.set_index('name').loc[input_channels]['channel_id'].values
		n_channels = self.input_ids.shape[0]

		output_channel = self.tfds_param['output_channels']
		self.output_id = channels.set_index('name').loc[output_channel]['channel_id'].values[0]

		# Add info to the description
		global _DESCRIPTION
		_DESCRIPTION += '\ninput_channels:\n{}'.format(input_channels)
		_DESCRIPTION += '\n\noutput_channel:\n{}'.format(output_channel)
		_DESCRIPTION += '\n\nTFDS metadata path:\n{}'.format(self.script_output_path)

		# Save output of this script
		self._save_tfds_metadata()

		return tfds.core.DatasetInfo(
			builder=self,
			description=_DESCRIPTION,
			features=tfds.features.FeaturesDict({
				# These are the features of your dataset like images, labels ...
				'mapobject_id_cell': tfds.features.Text(),
				'image': tfds.features.Tensor(shape=(img_size, img_size, n_channels), dtype=img_dtype),
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

		for fn in file_names:
			cell = np.load(images_path.joinpath(fn))
			cell_id = int(fn.split('.')[0])
			cell_img = cell['img'][:,:,self.input_ids]
			cell_target = [cell['targets'][self.output_id]]

			yield cell_id, {
				'mapobject_id_cell': str(cell_id),
				'image': cell_img,
				'target': cell_target,
				}

	def _save_tfds_metadata(self):

		# save original metadata:
		with open(os.path.join(self.script_output_path, 'metadata.csv'), 'w') as file:
			self.full_metadata.to_csv(file, index=False, sep=',')
		# Save filtered metadata
		with open(os.path.join(self.script_output_path, 'filtered_metadata.csv'), 'w') as file:
			self.filtered_metadata.to_csv(file, index=False, sep=',')
		# save tfds creation parameters
		with open(os.path.join(self.script_output_path, 'tfds_parameters.json'), 'w') as file:
			json.dump(self.tfds_param, file, indent=4)
		# save data preprocessing parameters
		with open(os.path.join(self.script_output_path, 'data_pp_parameters.json'), 'w') as file:
			json.dump(self.pp_param, file, indent=4)
