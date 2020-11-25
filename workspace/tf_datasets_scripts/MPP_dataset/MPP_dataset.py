"""MPP_dataset dataset."""

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

class MppDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for MPP_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    # Load tf dataset parameters
    # Path where the parameters to create the tf dataset are
    tf_param_path = './Parameters'
    with open(os.path.join(tf_param_path, 'tf_dataset_parameters.json')) as pp_file:
        tf_param = json.load(pp_file)

    # Path where the preprocessed data to be transformed into tf dataset is
    self.data_source_path = tf_param['data_source_path']

    # Load channel file:
    with open(os.path.join(self.data_source_path, 'channels.csv')) as channel_file:
        channels =  pd.read_csv(channel_file)
    # Load preprocessing parameters
    with open(os.path.join(self.data_source_path, 'params.json')) as pp_file:
        pp_param = json.load(pp_file)

    # Get info from the preprocessing parameters
    img_size = pp_param['img_size']

    input_channels = tf_param['input_channels']
    self.input_ids = channels.set_index('name').loc[input_channels]['channel_id'].values
    n_channels = self.input_ids.shape[0]

    output_channel = tf_param['output_channels']
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
            'image': tfds.features.Tensor(shape=(img_size, img_size, n_channels), dtype=tf.float64),
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
            images_path = input_data_path / 'train',
        ),
        'validation': self._generate_examples(
            images_path = input_data_path / 'val',
        ),
        'test': self._generate_examples(
            images_path = input_data_path / 'test',
        ),
    }

  def _generate_examples(self, images_path):
      """
      What is yield?
      https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
      generates the examples for each split from the source data.
      Yields examples.
      """

      for f in images_path.glob('*.npz'):
          cell = np.load(f)
          cell_id = int(f.name.split('.')[0])
          cell_img = cell['img'][:,:,self.input_ids]
          cell_target = [cell['targets'][self.output_id]]

          yield cell_id, {
            'mapobject_id_cell': str(cell_id),
            'image': cell_img,
            'target': cell_target,
            }
