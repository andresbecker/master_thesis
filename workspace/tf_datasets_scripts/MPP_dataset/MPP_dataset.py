"""MPP_dataset dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import tensorflow as tf

# TODO(MPP_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(MPP_dataset): BibTeX citation
_CITATION = """
"""


class MppDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for MPP_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(MPP_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'mapobject_id_cell': tfds.features.Text(),
            #'image': tfds.features.Image(shape=(224, 224, 38)),
            'image': tfds.features.Tensor(shape=(224, 224, 38), dtype=tf.float64),
            'target': tfds.features.Tensor(shape=(1,), dtype=tf.float64),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=None,
        supervised_keys=('image', 'target'),  # Set to `None` to disable
        #homepage='https://dataset-homepage/',
        #citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """
    downloads and splits data
    Returns SplitGenerators.
    This dataset was created following the guide:
    https://www.tensorflow.org/datasets/add_dataset
    """
    # TODO(MPP_dataset): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')
    preprocessed_data_path = '/home/hhughes/Documents/Master_Thesis/Project/datasets/184A1_hannah_imgs_scalars_test'
    #https://www.tensorflow.org/datasets/api_docs/python/tfds/download/DownloadManager
    extracted_path = dl_manager.extract(preprocessed_data_path)

    # TODO(MPP_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(
            images_path = extracted_path / 'test',
            metadata_path = extracted_path / 'metadata.csv',
        ),
        'validation': self._generate_examples(
            images_path = extracted_path / 'val',
            metadata_path = extracted_path / 'metadata.csv',
        ),
    }

  def _generate_examples(self, images_path, metadata_path):
    """
    What is yield?
    https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    generates the examples for each split from the source data.
    Yields examples.
    """

    #metadata = pd.read_csv(metadata_path)

    # TODO(MPP_dataset): Yields (key, example) tuples from the dataset
    for f in images_path.glob('*.npz'):

      cell_id = int(f.name.split('.')[0])
      #mask = (metadata.mapobject_id_cell == int(cell_id))
      #tr = metadata['00_EU_avg'][mask].values
      mpp_img = np.load(f)
      #print('sssssssssssssssssssssssss', mpp_img['img'].shape, 'sssssssssssssss')
      #print('sssssssssssssssssssssssss', mpp_img['targets'][35], 'sssssssssssssss')
      yield cell_id, {
          'mapobject_id_cell': str(cell_id),
          'image': mpp_img['img'],
          'target': [mpp_img['targets'][35]],
      }
