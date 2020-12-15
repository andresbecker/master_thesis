from pelkmans.bounding_box import BoundingBox
import os
import logging
import numpy as np
import pandas as pd
import json

class MPPData:
    def __init__(self, metadata, channels, labels, x, y, mpp, mapobject_ids, mcu_ids=None, conditions=None, seed=42):
        self.log = logging.getLogger(self.__class__.__name__)
        np.random.seed(seed)
        self.mapobject_ids = mapobject_ids
        self.log.info('creating new MPPData with {}'.format(self.mapobject_ids))
        # subset metadata to mapobbject_ids
        self.metadata = metadata[metadata.mapobject_id.isin(np.unique(self.mapobject_ids))]
        # leave only unique elements
        self.metadata = self.metadata.groupby(['mapobject_id_cell'], sort=False, as_index=False).first()
        self.channels = channels
        self.labels = labels
        self.x = x
        self.y = y
        self.seed = seed
        if len(mpp.shape) == 2:
            self.mpp = mpp[:,np.newaxis,np.newaxis,:]
        else:
            self.mpp = mpp
        if mcu_ids is None:
            mcu_ids = np.zeros(len(self.mapobject_ids), dtype=np.int64)
        self.mcu_ids = mcu_ids
        if conditions is None:
            conditions = np.zeros([len(self.mapobject_ids),1],  dtype=np.int64)
        self.conditions = conditions

    @property
    def has_neighbor_data(self):
        return (self.mpp.shape[1]!=1) and (self.mpp.shape[2]!=1)

    @property
    def center_mpp(self):
        c = self.mpp.shape[1]//2
        #print(self.mpp.shape)
        #print(c)
        return self.mpp[:,c,c,:]

    def __str__(self):
        return 'MPPData ({} mpps with shape {} from {} objects)'.format(self.mpp.shape[0], self.mpp.shape[1:], len(self.metadata))

    @classmethod
    def from_data_dir(cls, data_dir, dir_type='hannah', seed=42):
        # read all data from data_dir
        if dir_type == 'scott':
            well = data_dir.split('/')[-1]
            metadata = pd.read_csv(os.path.join(data_dir, '../METADATA/201908-NascentRNA-4i_cyc0-22_plate01_{}_Cells_metadata.csv'.format(well)))
        elif dir_type == 'hannah':
            metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'), index_col=0)
        channels = pd.read_csv(os.path.join(data_dir, 'channels.csv'), names=['channel_id', 'name'])
        x = np.load(os.path.join(data_dir, 'x.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        try:
            labels = np.load(os.path.join(data_dir, 'labels.npy'))
        except FileNotFoundError as e:
            labels = np.zeros(len(x), dtype=np.uint16)
        mpp = np.load(os.path.join(data_dir, 'mpp.npy'))
        mapobject_ids = np.load(os.path.join(data_dir, 'mapobject_ids.npy'))
        # init self
        self = cls(metadata=metadata, channels=channels, labels=labels, x=x, y=y, mpp=mpp, mapobject_ids=mapobject_ids, seed=seed)
        self.data_dir = data_dir
        return self

    @classmethod
    def concat(cls, objs):
        """concatenate the mpp_data objects by concatenating all arrays and return a new one"""
        # channels need to be the same
        for mpp_data in objs:
            assert (mpp_data.channels.name == objs[0].channels.name).all()
        channels = objs[0].channels
        # concatenate metadata (pandas)
        metadata = pd.concat([mpp_data.metadata for mpp_data in objs], axis=0, ignore_index=True)
        # concatenate numpy arrays
        labels = np.concatenate([mpp_data.labels for mpp_data in objs], axis=0)
        x = np.concatenate([mpp_data.x for mpp_data in objs], axis=0)
        y = np.concatenate([mpp_data.y for mpp_data in objs], axis=0)
        mpp = np.concatenate([mpp_data.mpp for mpp_data in objs], axis=0)
        mapobject_ids = np.concatenate([mpp_data.mapobject_ids for mpp_data in objs], axis=0)
        conditions = np.concatenate([mpp_data.conditions for mpp_data in objs], axis=0)
        #print(mapobject_ids)
        #for mpp_data in objs:
        #    print('\t', mpp_data.mapobject_ids)
        mcu_ids = np.concatenate([mpp_data.mcu_ids for mpp_data in objs], axis=0)
        self = cls(metadata=metadata, channels=channels, labels=labels, x=x, y=y, mpp=mpp, mapobject_ids=mapobject_ids, mcu_ids=mcu_ids,
                  conditions=conditions)
        self.log.info('Concatenated several MPPDatas')
        return self

    def merge_instances(self, objs):
        """
        Merge self variables with instances of same class (objs vars).
        This method is very similar to classmethod concat. The difference is
        that this modify the self instance instead of returning a new one.
        The aim of this method is to add support to images and masks.
        Input: List containing instances of class MPPData.
        Output: None.
        """
        for mpp_data in objs:
            if not all(mpp_data.channels.name == self.channels.name):
                raise Exception('Channels across MPPData instances are not the same!')
            if (vars(mpp_data).keys() != vars(self).keys()):
                raise Exception('Variables across MPPData instances are not the same!')

        # concatenate metadata (pandas)
        self.metadata = pd.concat([self.metadata]+[mpp_data.metadata for mpp_data in objs], axis=0, ignore_index=True)

        # Concatenate instances variables
        instance_vars = {'labels', 'x', 'y', 'mpp', 'mapobject_ids', 'mcu_ids','conditions', 'images', 'images_masks'}
        for var in set(vars(self).keys()).intersection(instance_vars):
            temp_var = np.concatenate([getattr(self, var)]+[getattr(mpp_data,var) for mpp_data in objs], axis=0)
            setattr(self, var, temp_var)

        self.log.info('Concatenated several MPPDatas')

    def filter_cells(self, filter_criteria=['is_border_cell'], filter_values=[0]):
        """
        Filter cells given the desired criteria (from metadata).
        Input:
            -filter_criteria: list containing the metadata column
             names
            -filter_values: list with values to be avoided (cutted)
             corresponding to filter_criteria entrances
        Output:
            modify self atributes, so metadata lables, x, y,
            mapobject_ids, mpp, conditions and mcu_ids only have cell
            information that fulfill the given parameters
        """

        msg = 'Starting filtering process with filters:\n{}\n{}'.format(filter_criteria, filter_values)
        self.log.info(msg)

        if len(filter_criteria) != len(filter_values):
            msg = 'length of filter_criteria and filter_values defined in input parameters does not match!'
            self.log.error(msg)
            raise Exception(msg)

        metadata_mask = np.ones(self.metadata.shape[0]).astype(bool)
        msg = 'Total number of cells: {}'.format(int(np.sum(metadata_mask)))
        self.log.info(msg)
        print(msg)
        for f, f_val in zip(filter_criteria, filter_values):
            if (f_val == 'nan') or (f_val == 'NaN') or (f_val == 'NAN'):
                mask_temp = ~self.metadata[f].isna().values
            else:
                mask_temp = ~(self.metadata[f] == f_val).values

            msg = '{} cells cutted by filter: {} == {}'.format(self.metadata.shape[0]-np.sum(mask_temp), f, f_val)
            self.log.info(msg)
            print(msg)

            metadata_mask &= mask_temp
        msg = 'Number of cutted cells: {}'.format(int(self.metadata.shape[0] - np.sum(metadata_mask)))
        self.log.info(msg)
        print(msg)

        # Filter metadata
        self.metadata = self.metadata.iloc[metadata_mask]

        # Get mapobject_ids from metadata that fulfill the given conditions
        mapobject_ids = self.metadata.mapobject_id.values
        # Get mask and filter lables, x, y, mapobject_ids, mpp, conditions and mcu_ids
        mask = np.in1d(self.mapobject_ids, mapobject_ids)

        instance_vars = {'labels', 'x', 'y', 'mpp', 'mapobject_ids', 'mcu_ids','conditions'}
        for var in instance_vars:
            setattr(self, var, getattr(self, var)[mask])

    def train_val_test_split(self, train_frac=0.8, val_frac=0.1):
        """split along mapobject_ids for train/val/test split"""
        ids = np.unique(self.mapobject_ids)
        np.random.seed(self.seed)
        np.random.shuffle(ids)
        num_train = int(len(ids)*train_frac)
        num_val = int(len(ids)*val_frac)
        train_ids = ids[:num_train]
        val_ids = ids[num_train:num_train+num_val]
        test_ids = ids[num_train+num_val:]
        self.log.info('splitting data in {} train, {} val, and {} test objects'.format(len(train_ids), len(val_ids), len(test_ids)))
        splits = []
        for split_ids in (train_ids, val_ids, test_ids):
            #ind = []
            #for cur_id in split_ids:
            #    ind.append(np.where(self.mapobject_ids==cur_id)[0])
            #ind = np.concatenate(ind, axis=0)
            ind = np.in1d(self.mapobject_ids, split_ids)
            splits.append(MPPData(metadata=self.metadata, channels=self.channels, labels=self.labels[ind],
                                  x=self.x[ind], y=self.y[ind], mpp=self.mpp[ind],
                                  mapobject_ids=self.mapobject_ids[ind], mcu_ids=self.mcu_ids[ind],
                                  conditions=self.conditions[ind]))
        return splits

    def add_cell_cycle_to_metadata(self, cc_file):
        """
        Add Cell cycle information to metadata.
        Input: Absolute path to cell cycle file
        Output: self.metadata with cell cycle information
        """

        msg = 'Adding Cell Cycle to metadata using file:\n{}'.format(cc_file)
        self.log.info(msg)

        if not os.path.exists(cc_file):
            msg = 'Cell cycle file {} not found!'.format(cc_file)
            self.log.error(msg)
            raise Exception(msg)

        cc_data = pd.read_csv(cc_file)
        self.metadata = self.metadata.merge(cc_data,
                                    left_on='mapobject_id_cell',
                                    right_on='mapobject_id',
                                    how='left',
                                    suffixes=('','_cc'))
        self.metadata = self.metadata.drop(['mapobject_id_cc'], axis=1)

    def add_well_info_to_metadata(self, well_file):
        """
        Add well information (cell line, perturbation and duration)
        to metadata.
        Input: Absolute path to well information file
        Output: self.metadata with well information
        """

        msg = 'Adding Well info to metadata using file:\n{}'.format(well_file)
        self.log.info(msg)

        if not os.path.exists(well_file):
            msg = 'Well metadata file {} not found!'.format(well_file)
            self.log.error(msg)
            raise Exception(msg)

        well_data = pd.read_csv(well_file)
        well_data = well_data[['plate_name', 'well_name', 'cell_type', 'perturbation', 'duration']]

        # Check for rows in well_data df with same plate_name and well_name values
        if (np.sum(well_data.groupby(['plate_name', 'well_name']).size().values > 1) > 0):
            msg = 'More than one row in {} with same combination of plate_name and well_name values!'.format(well_file)
            self.log.error(msg)
            raise Exception(msg)

        self.metadata = self.metadata.merge(well_data,
             left_on=['plate_name_cell', 'well_name_cell'],
             right_on=['plate_name', 'well_name'],
             how='left',
             suffixes=('','_wmd'))

        self.metadata = self.metadata.drop(['plate_name_wmd','well_name_wmd'], axis=1)

    def add_image_and_mask(self, data='MPP', remove_original_data=False, channel_ids=None, img_size=None, pad=0):
        """
        Add images and its respective mask to instance vars as numpy arrays.
        IMPORTANT: Images are stored as np.uint16 which means that it can only hold values in [0,65535] (which is the range values for MPPData). This allows to store images without using tons of ram memory. Therefore, this method should be run BEFORE normalizing the original data! Otherwise most of the data will be set to 0!
        The normalization of the images can be done only during saving into disk. To do it, after running this method run the function get_image_normalization_vals to get the normalizing values per channel, and then run the function normalize_and_save_MPPData_images to normalize them during saving.
        TODO: Implement support for data different than 'MPP'
        Input:
            data: str indicating data type
            channel_ids: 1D array indicating id channels to be contemplated in the returned image and mask
            img_size: Natural Number, size for output images (i.e. shape: (img_size,img_size))
            pad: amount of padding added to returned image (only used when img_size is None)
            remove_original_data: boolean indicating if data used to create the images and masks should be deletad after the convertion.
        Output (added to self):
            -imgs: array of shape (n_observations,img_size,img_size,len(channel_ids))
            -mask: boolean array of same shape as imgs. Array entrance = True
                if value came from MPPData.
        """

        msg = 'Adding images and masks to MPPData'
        self.log.info(msg)

        imgs = []
        mask = []
        for mapobject_id in self.metadata.mapobject_id:
            if data == 'MPP':
                res = self.get_mpp_img(mapobject_id, channel_ids, img_size=img_size, pad=pad)
                res_m = self.get_mpp_img(mapobject_id, get_mask=True, img_size=img_size, pad=pad).astype(np.bool).reshape(res.shape[:-1])
            else:
                msg = 'Data type different that MPP given! Not implemented yet!'
                self.error.info(msg)
                raise NotImplementedError
            if img_size is None:
                res = res[0]
                res_m = res_m[0]
            imgs.append(res)
            mask.append(res_m)

        if remove_original_data:
            msg = 'Deleting origin MPPData...'
            self.log.info(msg)
            print(msg)
            del(self.labels, self.x, self.y, self.mpp, self.mapobject_ids, self.mcu_ids, self.conditions)

        self.images = np.array(imgs).astype(np.uint16)
        self.images_masks = np.array(mask)

    def add_scalar_projection(self, method='avg'):
        """
        This method projects each cell (and each one of its channels) into a scalar. For instance, assuming images, if one have data.images.shape = (100,240,240, 38) (100 images of size 240x240 and 38 channels), then this method projects data.images into an array of shape (100, 38).
        Input:
            -method: String indicating the used function to project each image into a single number. Available functions are the average ('avg') and the median ('median').
        Output: No output. For instance, if 'avg' selected, then m columns are added to self.metadata containing the average number of each cell and channel (m represents the number of channels).
        """
        n_cells = self.metadata.shape[0]
        n_channels = self.mpp.shape[-1]
        cell_ids = np.array(self.metadata.mapobject_id.values)

        col_name = ['mapobject_id', 'cell_size']
        if (method == 'size_and_sum'):
            col_name += [self.channels.set_index('channel_id').loc[c].values[0]+'_sum' for c in range(n_channels)]
            col_name += [self.channels.set_index('channel_id').loc[c].values[0]+'_size' for c in range(n_channels)]
        else:
            col_name += [self.channels.set_index('channel_id').loc[c].values[0]+'_'+method for c in range(n_channels)]

        scalar_vals_df = pd.DataFrame(columns=col_name)

        for map_id in cell_ids:
            mask = (self.mapobject_ids == map_id)
            cell_size = self.mpp[mask].shape[0]
            temp_data = np.array([map_id, cell_size])

            if (method == 'avg'):
                channel_sclara = self.mpp[mask].mean(axis=0).reshape(-1)

            elif (method == 'median'):
                channel_sclara = np.median(self.mpp[mask], axis=0).reshape(-1)

            elif (method == 'size_and_sum'):
                channel_sum = self.mpp[mask].sum(axis=0).reshape(-1)
                channel_size = list((self.mpp[mask] > 0).sum(axis=0).reshape(-1))
                channel_sclara = np.concatenate((channel_sum, channel_size), axis=0)

            else:
                print('Available methods:\n avg, median')
                raise NotImplementedError(method)

            channel_sclara = np.concatenate((np.array([map_id, cell_size]), channel_sclara), axis=0).reshape((1,-1))
            temp_df = pd.DataFrame(channel_sclara, columns=col_name)
            scalar_vals_df = scalar_vals_df.append(temp_df, ignore_index=True)

        scalar_vals_df.mapobject_id = scalar_vals_df.mapobject_id.astype('uint32')
        scalar_vals_df = scalar_vals_df.set_index(['mapobject_id'])

        # Merge scalar data with metadata
        self.metadata = self.metadata.merge(
                                scalar_vals_df,
                                left_on='mapobject_id',
                                right_on='mapobject_id',
                                how='left',
                                suffixes=('', 'ss'))

    def subtract_background(self,background_value):
        # code copied/adapted from Scott Berry's MCU package
        # Note mpp is converted to float
        if isinstance(background_value, float):
            self.log.info('subtracting constant value of {} from all channels'.format(background_value))
            self.mpp == self.mpp.astype(np.float32) - background_value
        else:
            self.log.debug('reading channel-specific background from {}'.format(background_value))
            bkgrd = pd.read_csv(background_value).merge(self.channels, left_on='channel', right_on='name', how='right')
            bkgrd = bkgrd.loc[bkgrd['measurement_type'] == "non-cell"].loc[bkgrd['cell_line']== "HeLa"]
            bkgrd = bkgrd[['channel_id','mean_background']]

            # create a dictionary to link mpp columns with their background values
            bkgrd_dict = bkgrd.set_index('channel_id')['mean_background'].to_dict()
            # check all channels are present
            if len(bkgrd_dict) != self.channels.shape[0]:
                missing_channels = list(set(self.channels.channel_id).difference(bkgrd_dict.keys()))
                self.log.warning('missing background value for channels {}'.format(list(self.channels.loc[missing_channels].name)))

            # subtract per-channel background (row-wise) from mpp
            # if background not knows, subtract 0
            bkgrd_vec = np.array(
                [bkgrd_dict.get(i, 0) for i in range(0,self.channels.shape[0])]).astype(np.float64)

            self.log.info('subtracting channel-specific background: {}'.format(
                ', '.join([str(el) for el in bkgrd_vec.tolist()])
            ))
            self.mpp = self.mpp.astype(np.float64) - bkgrd_vec
        # cut off at 0 (no negative values)
        self.mpp[self.mpp<0] = 0

    def rescale_intensities_per_channel(self,percentile=98.0,rescale_values=None):
        # Note mpp is modified in place and function only returns the
        # normalization values

        if rescale_values is None:
            if 'mpp' in vars(self).keys():
                rescale_values = np.percentile(self.center_mpp, percentile, axis=0)
            elif {'images', 'images_masks'}.issubset(vars(self).keys()):
                n_channels =  self.images.shape[-1]
                rescale_values = []
                for c in range(n_channels):
                    rescale_values.append(np.percentile(self.images[:,:,:,c][self.images_masks], percentile, axis=0))
                rescale_values = np.array(rescale_values)
            else:
                raise Exception('Not enough information to calculate the rescale values')
        self.log.info('rescaling mpp intensities per channel with values {}'.format(rescale_values))

        if 'mpp' in vars(self).keys():
            self.mpp /= rescale_values
            self.log.info('converting mpp values to float32')
            self.mpp = self.mpp.astype(np.float32)

        if 'images' in vars(self).keys():
            imgs_shape = self.images.shape
            self.images = self.images.reshape(-1,imgs_shape[-1]) / rescale_values
            self.images = self.images.reshape(imgs_shape)
            self.log.info('converting mpp values to float32')
            self.images = self.images.astype(np.float32)

        return rescale_values

    def get_img_from_data(self, x, y, data, img_size=None, pad=0):
        """
        Create image from x and y coordinates and fill with data.
        Args:
            x, y: 1-d array of x and y corrdinates
            data: shape (n_coords,n_channels), data for the image
            img_size: size of returned image
            pad: amount of padding added to returned image (only used when img_size is None)
        """
        x_coord = x - x.min() + pad
        y_coord = y - y.min() + pad
        # create image
        img = np.zeros((y_coord.max()+1+pad, x_coord.max()+1+pad, data.shape[-1]), dtype=data.dtype)
        img[y_coord,x_coord] = data
        # resize
        if img_size is not None:
            c = BoundingBox(0,0,img.shape[0], img.shape[1]).center
            bbox = BoundingBox.from_center(c[0], c[1], img_size, img_size)
            img = bbox.crop(img)
            return img
        else:
            # padding info is only relevant when not cropping img to shape
            return img, (x.min()-pad, y.min()-pad)

    def get_mpp_img(self, mapobject_id, channel_ids=None, get_mask=False, **kwargs):
        """
        Calculate MPP image (or mask if get_mask=True) of given mapobject.
        Input:
            mapobject_ids: mapobject_id (integer) corresponding to the cell that will be transformed into image
            channel_ids: list containing the ids of MPP channels that the image should have. If None, all channels are returned.
            get_mask: boolean. Indicates if mask or image corresponding to mapobject_id should be returned.
        Output:
            If get_mask is False, then get_mpp_img returns the corresponding image to mapobject_id.
            If get_mask is True, then get_mpp_img returns a mask of same size of the corresponding image to mapobject_id, indicating wich pixels in the image were measured in the original data.
        """
        if channel_ids is None:
            channel_ids = range(len(self.channels))
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        if get_mask:
            data = np.ones((self.center_mpp[mask][:,0:1]).shape, dtype=np.short)
        else:
            data = self.center_mpp[mask][:,channel_ids]

        return self.get_img_from_data(x, y, data, **kwargs)

    def get_object_imgs(self, data='MPP', channel_ids=None, img_size=None, pad=0):
        """
        Return images from differen data types.
        Input:
            data: string indicating data type. Note: MPP-mask returns the corresponding masks to MPP data that indicates which pixels were measured in the origin data and which ones were added to construct the image.
            channel_ids: channel ids to be included in the output images.
            img_size: integer to indicat size of output images (shape:(n_cels,img_size,img_size,len(channel_ids)))
            pad: integer indicatting number of 0 rows and columns around the original image.
        Output:
        """
        imgs = []
        for mapobject_id in self.metadata.mapobject_id:
            if data == 'MPP':
                res = self.get_mpp_img(mapobject_id, channel_ids, img_size=img_size, pad=pad)
            elif data == 'MPP-mask':
                res = self.get_mpp_img(mapobject_id, get_mask=True, img_size=img_size, pad=pad).astype(np.bool)
                res = res.reshape(res.shape[:-1])
            else:
                raise NotImplementedError
            if img_size is None:
                res = res[0]
            imgs.append(res)
        return imgs

def get_image_normalization_vals(instance_dict=None, input_channel_ids=None, percentile=98):
    """
    This function calculate the percentiles for each channel of a group of MPPData images. The main advantage of this function is that it calculates the percentile from data in separated arrays.
    Input:
        instance_dict: dictionary containing MPPData instances WITH attributes images and images_masks.
        input_channel_ids: list indicating for which channels we must calculate the normalization parameters.
        percentile: integer in [0,100].
    Output:
        rescale_values: numpy array of lenght number_of_channels, containing the normalization values of each channel
    """

    log = logging.getLogger()
    msg = 'Starting computation of normalization values (from mpp images and masks).'
    log.info(msg)

    n_channels = instance_dict[0].images.shape[-1]
    if input_channel_ids is None:
        input_channel_ids = range(n_channels)
    else:
        n_channels = len(input_channel_ids)

    # Since there is no wey to concatenate np arrays in place (without creating a copy), we first create an array that will be filled with the values necessary to calculate the percentile per channel. Note that the size of this array is the same for every channel.
    #https://stackoverflow.com/questions/7869095/concatenate-numpy-arrays-without-copying
    n_pixels = 0
    for well in instance_dict:
        n_pixels += well.images_masks.sum()
    msg = 'Number of pixels for each channel and for all the images:{}'.format(n_pixels)
    log.info(msg)

    # For each channel we gather the values from all the images to calculete the percentile
    rescale_values = []
    for c in input_channel_ids:
        msg = 'Calculating percentil {} for channel {}/{}...'.format(percentile, c, n_channels)
        log.info(msg)
        #print(msg)
        idx_l = 0
        idx_h = 0
        channel_pixels = np.zeros(n_pixels, dtype=np.uint16)
        for well in instance_dict:
            idx_l = idx_h
            idx_h += well.images_masks.sum()
            channel_pixels[idx_l:idx_h] = well.images[:,:,:,c][well.images_masks]

        rescale_values.append(np.percentile(channel_pixels, percentile, axis=0))
    rescale_values = np.array(rescale_values)

    msg = 'Computation of normalization values finished!'
    log.info(msg)
    print(msg)
    msg = 'Normalization values:\n{}'.format(rescale_values)
    log.info(msg)

    return rescale_values

def save_to_file_targets_masks_and_normalized_images(mppdata_dict=None, norm_vals=None, channels_ids=None, projection_method='avg', outdir=None, dtype=np.float32):
    """
    This function take a dictionary (mppdata_dict), where each entrance is a list with MPPData objects, and save each image, mask and target values in a singel npz file using the mapobject_id_cell as name (i.e. one npz file per cell).
    The normalization is also made in this function to avoid duplicating np arrays in ram memory during this process. Another advatnage is that this allow us to save in ram memory MPPData.images as np.uint16 (integer in [0, 65535], which is the range of values for MPPData) during data processing, reducing signifacntly the necessary ram memory.
    Beside the image and its mask, the npz file also contain a vector call 'targets' which has the projection of each channel into a scalar.
    Input:
        mppdata_dict: dictionary containing lists contaning instances of the class MPPData.
        norm_vals: np array containing the normalizing value of each channel.
        channels_ids: np array containing the ids of the channels to be saved.
        projection_method: Method used to project each image channel into a scalar. The implemented ones are average 'avg' and 'median'.
        outdir: directory to save the processed data
    Output:
        output_paths: dictionary containing the file paths for each key in mppdata_dict where images and mask were saved.
    """
    log = logging.getLogger()

    # Create directories to save images
    log.info('Creating directories to save data from instances...')
    output_paths = {}
    for dir_names in list(mppdata_dict.keys()):
        path_temp = os.path.join(outdir, dir_names)
        output_paths[dir_names] = path_temp
        if os.path.exists(path_temp):
            # Remove previous files
            for f in os.listdir(path_temp):
                os.remove(os.path.join(path_temp,f))
        else:
            os.makedirs(path_temp, exist_ok=False)
    log.info('Directories created:\n{}'.format(output_paths))

    k = list(mppdata_dict.keys())[0]
    n_channels = mppdata_dict[k][0].images.shape[-1]
    if channels_ids is None:
        channels_ids = np.arange(n_channels)
    else:
        n_channels = channels_ids.shape[0]
    img_shape = mppdata_dict[k][0].images[:,:,:,channels_ids].shape[1:]

    # If normalization values are not given, then we devide by 1
    if norm_vals is None:
        norm_vals = np.ones(n_channels).astype(dtype)
    else:
        norm_vals = norm_vals.astype(dtype)

    # Iterate over train, val and test lists of instances
    for key in mppdata_dict.keys():
        msg = 'Saving {} images and masks...'.format(key)
        log.info(msg)
        print(msg)
        # iterate over each mppdata instances in e.g. train list
        for i, mppdata in enumerate(mppdata_dict[key]):
            n_cells = mppdata.metadata.shape[0]
            msg = 'Saving {} images form {}-well {}/{}...'.format(n_cells,key, i, len(mppdata_dict[key]))
            log.info(msg)
            print(msg)
            for j in range(n_cells):
                # Create file name
                cell_id = mppdata.metadata.iloc[j]['mapobject_id_cell']
                file_name = str(cell_id)+'.npz'
                file_name = os.path.join(output_paths[key], file_name)

                # Get image and mask to create projection and save
                temp_mask = mppdata.images_masks[j]
                temp_img = mppdata.images[j][:,:,channels_ids].copy()

                # Project each channel to a singel number (avg) to save them as a vector of targets
                # Note that the target value is calculated without Normalization!
                targets = []
                for c in channels_ids:
                    if (projection_method == 'avg'):
                        temp_target = temp_img[:,:,c][temp_mask].mean()

                    elif (projection_method == 'size_and_sum'):
                        temp_target = ((temp_img[:,:,c] > 0).sum(), temp_img[:,:,c].sum())

                    elif (projection_method == 'median'):
                        temp_target = np.median(temp_img[:,:,c][temp_mask], axis=0)

                    else:
                        msg = 'Projection method {} not implemented yet!'.format(projection_method)
                        log.error(msg)
                        raise NotImplementedError(msg)
                    targets.append(temp_target)
                targets = np.asarray(targets).astype(dtype)

                temp_img = temp_img.reshape((-1, n_channels))
                temp_img = temp_img.astype(dtype)
                temp_img /= norm_vals[channels_ids]
                temp_img = temp_img.reshape(img_shape)

                # Save everything
                np.savez(file_name, img=temp_img, mask=temp_mask, targets=targets)

    msg = 'MPPData images and masks saving process finished!'
    log.info(msg)
    print(msg)

    return output_paths

def get_concatenated_metadata(mppdata_dict=None, normalize=True, norm_key='train', projection_method='avg', percentile=98):
    """
    Concatenate (and normalize projected data, if normalize=True) metadata of MPPData instances given.
    Input:
        mppdata_dict: dictionary containing lists contaning instances of the class MPPData
        normalize: boolean. If true, then normalization of projected data is done.
        norm_key: string indicating the mppdata_dict key that contains the data to calculate the normalization values (usaually 'train' or 'training').
        projection_method: method used to project each mpp image channel into a scalar.
        percentile: integer in [0,100].
    Output:
        metadata: pandas df with the concatenated metadata in mppdata_dict
        normalization_vals: numpy array of lenght number_of_channels, containing the normalization values for each channel.
    """

    log = logging.getLogger()
    msg='Starting metadata concatenation process...'
    log.info(msg)

    # First, merge all metadata
    k = list(mppdata_dict.keys())[0]
    metadata_cols = list(mppdata_dict[k][0].metadata.columns)
    metadata = pd.DataFrame(columns=metadata_cols+['set'])

    for key in mppdata_dict.keys():
        for well in mppdata_dict[key]:
            temp_df = well.metadata.copy()
            temp_df['set'] = key
            metadata = pd.concat([metadata, temp_df],
                                 axis=0,
                                 ignore_index=True)

    if normalize:
        msg='Normalizing projected values in metadata...'
        log.info(msg)
        # Second, get normalization values
        if projection_method == 'size_and_sum':
            norm_columns = [c+'_sum' for c in mppdata_dict[norm_key][0].channels.name.values]
            norm_columns += [c+'_size' for c in mppdata_dict[norm_key][0].channels.name.values]
        else:
            norm_columns = [c+'_'+projection_method for c in mppdata_dict[norm_key][0].channels.name.values]
        normalization_vals =np.percentile(metadata[norm_columns][metadata.set == norm_key].values, percentile, axis=0)

        # Finally, normalize values
        for i, col in enumerate(norm_columns):
            metadata[col] /= normalization_vals[i]

        msg='Normalization values:\n{}'.format(normalization_vals)
        log.info(msg)

        return metadata, normalization_vals
    else:
        return metadata, 0
