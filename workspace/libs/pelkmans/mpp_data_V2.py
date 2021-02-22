from pelkmans.bounding_box import BoundingBox
import os
import logging
import numpy as np
import pandas as pd
import json
import tensorflow as tf

class MPPData:
    def __init__(self, metadata, channels, labels, x, y, mpp, mapobject_ids, mcu_ids=None, conditions=None):
        self.log = logging.getLogger(self.__class__.__name__)
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
    def from_data_dir(cls, data_dir, dir_type='hannah'):
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
        self = cls(metadata=metadata, channels=channels, labels=labels, x=x, y=y, mpp=mpp, mapobject_ids=mapobject_ids)
        self.data_dir = data_dir
        return self

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

    #@tf.function
    def _set_img_size(self, tensor_img=None, target_img_size=None, method='nearest'):
        """
        Upsample or downsample (ZoomIn or ZoomOut) and image
        tensor_img: tf tensor, multichannel image
        target_img_size: integer, desire image size
        method: string, interpolation method
            To see interpolation methods visit:
            https://www.tensorflow.org/api_docs/python/tf/image/ResizeMethod
            https://www.tensorflow.org/api_docs/python/tf/image/resize
        """
        # Get size of the image
        img_size = tensor_img.shape[1]

        tensor_img = tf.cast(tensor_img, dtype=tf.float32)

        if target_img_size != img_size:
            target_img_size = tf.cast(target_img_size, dtype=tf.int32)
            target_img_size = tf.repeat(target_img_size,2)
            tensor_img = tf.image.resize(tensor_img,
                                         size=target_img_size,
                                         method=method,
                                         preserve_aspect_ratio=False,
                                         antialias=True)

        return tensor_img

    def _get_cell_size(self, mapobject_id):
        """
        Return the size of a given cell (mapobject_id)
        """
        temp_mask = (self.mapobject_ids == mapobject_id)
        size_x = (self.x[temp_mask].max() - self.x[temp_mask].min())
        size_y = (self.y[temp_mask].max() - self.y[temp_mask].min())

        return np.max([size_x, size_y])

    def _get_cell_size_ratio(self, img_mask):
        """
        This function approximate the cell_size-img_size ratio, i.e. the ratio between how much of the image is being occupied by the cell information (the measured pixels) and how much by the added black pixels.
        """
        x_proyection = (np.sum(img_mask, axis=1) > 0)
        y_proyection = (np.sum(img_mask, axis=0) > 0)
        img_size = img_mask.shape[0]

        # min distance between the cell and the image border on the x axis
        x_distance = np.min((np.argmax(x_proyection), np.argmax(np.flip(x_proyection))))
        # min distance between the cell and the image border on the y axis
        y_distance = np.min((np.argmax(y_proyection), np.argmax(np.flip(y_proyection))))
        # min distance between the cell and the image border
        min_space = np.min((x_distance, y_distance))

        #cell-image ratio
        return 1 - 2 * (min_space / img_size)

    def save_img_mask_and_target_into_fs(self, outdir=None, data='MPP', input_channels=None, output_channels=None, projection_method='avg', img_size=224, pad=0, img_saving_mode='original_img_and_fixed_size', img_interpolation_method='nearest', dtype='float32', return_cell_size_ratio=False):
        """
        This function save into file system (fs) the extracted image cells, masks and targets from the current MPPData instance as individual files. Each file is named using the mapobject_id of the cell.
        This function also returns the cell size ratio.
        """

        self.log.info('Starting imgs, masks and targets saving process')

        if return_cell_size_ratio:
            cell_size_ratio_df = pd.DataFrame(columns=['mapobject_id_cell', 'cell_size_ratio'])

        # Process and save only filtered cells
        cell_ids = self.metadata[['mapobject_id_cell', 'mapobject_id']].values
        for (mapobject_id_cell, mapobject_id) in cell_ids:

            # Depending on img_save_mode, set the raw_img_size
            # save original cell img without fixed size or shape (it could be recangular)
            if img_saving_mode == 'original_img':
                raw_img_size = None
            # save original cell img without fixed size but fixed shape (squared)
            elif img_saving_mode == 'original_img_and_squared':
                # get the max size of the cell to get a squared img
                raw_img_size = self._get_cell_size(mapobject_id)
            # save original cell img with fixed size (squared)
            elif img_saving_mode == 'original_img_and_fixed_size':
                raw_img_size = img_size
            # save cell img with a fixed size (squared), it may need up or down sampling
            elif img_saving_mode == 'fixed_cell_size':
                # get the max size of the cell to get a squared img
                raw_img_size = self._get_cell_size(mapobject_id)

            # Get cell image and mask
            if data == 'MPP':
                img = self.get_mpp_img(mapobject_id, img_size=raw_img_size, pad=pad)
                img_mask = self.get_mpp_img(mapobject_id, get_mask=True, img_size=raw_img_size, pad=pad)
            else:
                msg = 'Data type different that MPP given! Not implemented yet!'
                self.error.info(msg)
                raise NotImplementedError
            if raw_img_size is None:
                img = img[0]
                img_mask = img_mask[0]

            # Get target values
            temp_mask = (self.mapobject_ids == mapobject_id)
            if (projection_method == 'avg'):
                channel_sclara = self.mpp[temp_mask].mean(axis=0).reshape(-1)
            elif (projection_method == 'median'):
                channel_sclara = np.median(self.mpp[temp_mask], axis=0).reshape(-1)
            else:
                msg = 'Available methods:\n avg, median'
                self.log.error(msg)
                print(msg)
                raise NotImplementedError(projection_method)

            # Apply upsampling or downsampling to the image and mask
            if img_saving_mode == 'fixed_cell_size':
                # if GPU available, disable it
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                # Resize image
                temp_tensor = tf.constant(img)
                temp_tensor = self._set_img_size(tensor_img=temp_tensor, target_img_size=img_size, method=img_interpolation_method)
                img = temp_tensor.numpy()
                # Resize mask
                temp_tensor = tf.constant(img_mask)
                temp_tensor = self._set_img_size(tensor_img=temp_tensor, target_img_size=img_size, method=img_interpolation_method)
                temp_mask = temp_tensor.numpy()
                # In case method != nearest, convert mask again to boolean
                img_mask = np.zeros(temp_mask.shape)
                img_mask[temp_mask >= 0.5] = 1

            if return_cell_size_ratio:
                cell_size_ratio = self._get_cell_size_ratio(img_mask)
                cell_size_ratio_df = cell_size_ratio_df.append({'mapobject_id_cell':mapobject_id_cell, 'cell_size_ratio':cell_size_ratio}, ignore_index=True)

            # Before saving, set I/O channel ids
            if input_channels is None:
                input_ids = range(img.shape[-1])
            else:
                input_ids = self.channels.set_index('name').loc[input_channels].channel_id.values
            if output_channels is None:
                output_ids = range(img.shape[-1])
            else:
                output_ids = self.channels.set_index('name').loc[output_channels].channel_id.values

            # Save img, mask and targets
            img = img[:,:,input_ids].astype(getattr(np, dtype))
            img_mask = img_mask.astype(np.bool).reshape(img.shape[:-1])
            targets = channel_sclara[output_ids]
            file_name = os.path.join(outdir, str(mapobject_id_cell)+'.npz')
            np.savez(file_name, img=img, mask=img_mask, targets=targets)

        if return_cell_size_ratio:
            cell_size_ratio_df['mapobject_id_cell'] = cell_size_ratio_df.mapobject_id_cell.astype('int64')
            return cell_size_ratio_df

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
        Calculate MPP image (or mask if get_mask=True) of a given mapobject.
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
