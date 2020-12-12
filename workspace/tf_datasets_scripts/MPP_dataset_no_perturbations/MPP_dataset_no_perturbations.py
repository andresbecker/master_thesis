"""MPP_dataset_no_perturbations dataset."""

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

This Dataset contains only the cells with no perturbations (i.e. cells such that perturbation in ['normal', 'DMSO']). Although perturbations 'TSA' seams not to have influence over the TR (00_EU avg), it seams to have an influence on the intensity of the channel 10_H3K27ac, and therefore it is not included.

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

class MppDataset_no_perturbations(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for MPP_dataset_no_perturbations dataset."""
	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
		'1.0.0': 'First Release. Only normal and DMSO perturbations included.',
		}

	def _info(self) -> tfds.core.DatasetInfo:
		"""Returns the dataset metadata."""
	  	# Load tf dataset parameters
	  	# Path where the parameters to create the tf dataset are
		tfds_param_path = './Parameters'
		with open(os.path.join(tfds_param_path, 'tf_dataset_parameters.json')) as file:
			tfds_param = json.load(file)

		# images data type
		img_dtype = getattr(tf, tfds_param['images_dtype'])

		# Path where the preprocessed data to be transformed into tf dataset is
		self.data_source_path = tfds_param['data_source_path']

		# Which perturbations to use?
		perturbations = tfds_param['perturbations']

		# Load metadata file:
		with open(os.path.join(self.data_source_path, 'metadata.csv')) as file:
			metadata =  pd.read_csv(file)
		# filter acordinglly to the selected perturbations
		metadata = metadata[metadata.perturbation.isin(perturbations)]
		# leave only unique elements
		metadata = metadata.groupby(['mapobject_id_cell'], sort=False, as_index=False).first()

		# filtered mapobject_id_cells
		self.filtered_cells = metadata.mapobject_id_cell.values.astype(str)

		# Load channel file:
		with open(os.path.join(self.data_source_path, 'channels.csv')) as file:
			channels =  pd.read_csv(file)

		# Load preprocessing parameters
		with open(os.path.join(self.data_source_path, 'params.json')) as file:
			pp_param = json.load(file)

		# Get info from the preprocessing parameters
		img_size = pp_param['img_size']

		# Get random seed
		try:
			self.seed = pp_param['seed']
		except:
			self.seed = 123

		input_channels = tfds_param['input_channels']
		self.input_ids = channels.set_index('name').loc[input_channels]['channel_id'].values
		n_channels = self.input_ids.shape[0]

		output_channel = tfds_param['output_channels']
		self.output_id = channels.set_index('name').loc[output_channel]['channel_id'].values[0]

		normalization_vals = pp_param['normalise_rescale_values']
		normalization_vals = np.asarray(normalization_vals)[self.input_ids]

		# Add info to the description
		global _DESCRIPTION
		_DESCRIPTION += '\ninput_channels:\n{}'.format(input_channels)
		_DESCRIPTION += '\n\noutput_channel:\n{}'.format(output_channel)
		_DESCRIPTION += '\n\nNormalization parameters:\n{}'.format(normalization_vals)

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

		return {
			'train': self._generate_examples(
				images_path = input_data_path / 'train'),
			'validation': self._generate_examples(
            	images_path = input_data_path / 'val'),
        	'test': self._generate_examples(
            	images_path = input_data_path / 'test'),
				}

	def _generate_examples(self, images_path):
		"""
	  	What is yield?
	  	https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	  	generates the examples for each split from the source data. Yields examples.
	  	"""

		# Get file names and shuffle them
		file_names = [file.name for file in images_path.glob('*.npz')]
		# filtered mapobject_id_cells
		file_names = [fn for fn in file_names if fn[:-4] in self.filtered_cells]
		file_names = np.array(file_names)

		# Shuffle files (not sure if this is necessary)
		np.random.seed(self.seed)
		np.random.shuffle(file_names)

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
