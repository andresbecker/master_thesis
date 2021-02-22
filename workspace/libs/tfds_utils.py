import tensorflow_datasets as tfds
import pandas as pd
import os
import json

METADATA_FILE_PREFIX = 'meta'

class Costum_TFDS_metadata(tfds.core.Metadata, dict):
    """
    A `tfds.core.Metadata` object that acts as a `dict`.
    Sub-class tfds.core.Metadata to overwrite tfds.core.Metadata methods save_metadata and load_metadata. The objective is to allows the TFDS to save additional general information about the dataset which are not specific to a feature or individual example. For more info visit:
    https://www.tensorflow.org/datasets/api_docs/python/tfds/core/Metadata
    """

    def save_metadata(self, data_dir):
        """
        Save TFDS extra info into disc in the same location where thr TFDS is saved.
        """

        for key in self.keys():
            # get type of data
            dt = key.split("_")[-1]
            # save parameters
            if dt == 'parameters':
                with open(os.path.join(data_dir, METADATA_FILE_PREFIX+'_'+key+'.json'), "w") as file:
                    json.dump(self[key], file, indent=4)
            # save dataframes
            elif dt == 'df':
                with open(os.path.join(data_dir, METADATA_FILE_PREFIX+'_'+key+'.csv'), "w") as file:
                    self[key].to_csv(file, index=False)
            else:
                raise NotImplementedError()

    def load_metadata(self, data_dir):
        """Restore the TFDS extra info. Load all supported files (.json and csv) that start with the prefix METADATA_FILE_PREFIX"""
        self.clear()

        # create temp dict to store loaded info
        temp_dict = {}
        # list available files and select only the ones starting with 'EI'
        files = [f for f in os.listdir(data_dir) if f.split("_")[0] == METADATA_FILE_PREFIX]
        for file in files:
            # get type of data
            dt = file.split(".")[-1]
            # get key to store info
            key = file.split(".")[:-1][0]
            key = key.split("_")[1:]
            key = "_".join(key)

            with open(os.path.join(data_dir, file), "r") as file:
                # open parameters
                if dt == 'json':
                    temp_dict[key] = json.load(file)
                # open dataframes
                elif dt == 'csv':
                    temp_dict[key] = pd.read_csv(file)
                else:
                    raise NotImplementedError()

        #self.fromkeys(['TFDS_metadata'], temp_dict)
        return temp_dict
