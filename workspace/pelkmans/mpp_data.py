from pelkmans.constants import DATA_DIR, convert_cell_cycle, convert_perturbations
from pelkmans.bounding_box import BoundingBox
import os
import logging
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import multiprocessing
from functools import partial
from numba import jit

class MPPDataset:
    def __init__(self, dataset_name, dataset_dir=os.path.join(DATA_DIR, 'datasets')):
        self.dataset_folder = os.path.join(dataset_dir, dataset_name)

        self.channels = pd.read_csv(os.path.join(self.dataset_folder, 'channels.csv'), index_col=0)
        self.channels = self.channels.reset_index().set_index('name')
        self.metadata = pd.read_csv(os.path.join(self.dataset_folder, 'metadata.csv'))

        # data
        self.data = {
            'train': np.load(os.path.join(self.dataset_folder, 'train_dataset.npz')),
            'val': np.load(os.path.join(self.dataset_folder, 'val_dataset.npz')),
            'test': np.load(os.path.join(self.dataset_folder, 'test_dataset.npz'))
        }
        self.imgs = {
            'val': np.load(os.path.join(self.dataset_folder, 'val_imgs.npz')),
            'test': np.load(os.path.join(self.dataset_folder, 'test_imgs.npz'))
        }

    def get_channel_ids(self, to_channels, from_channels=None):
        if from_channels is None:
            from_channels = self.channels
        if not isinstance(from_channels, pd.DataFrame):
            from_channels = pd.DataFrame({'name': from_channels, 'index': np.arange(len(from_channels))})
            from_channels = from_channels.set_index('name')
        from_channels = from_channels.reindex(to_channels)
        return list(from_channels['index'])

    def get_tf_dataset(self, split='train', output_channels=None, is_conditional=False, repeat_y=False):
        """returns tf.data.Dataset of the desired split"""
        x = self.data[split]['x']
        if is_conditional:
            x = (x, self.data[split]['c'])
        y = self.data[split]['y']
        if output_channels is not None:
            channel_ids = self.get_channel_ids(output_channels)
            y = y[:,channel_ids]
        if repeat_y is not False:
            y = tuple([y for _ in range(repeat_y)])
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        return dataset

    def get_y(self, split='train', output_channels=None):
        y = self.data[split]['y']
        if output_channels is not None:
            channel_ids = self.get_channel_ids(output_channels)
            y = y[:,channel_ids]
        return y

    def get_mapobject_ids(self, split='train', data='data'):
        if data == 'data':
            return self.data[split]['mapobject_id']
        else:
            return self.imgs[split]['mapobject_id']

    def get_imgs(self, split='val', img_ids=None, is_conditional=False):
        if img_ids is None:
            img_ids = np.arange(len(self.imgs[split]['img']))
        # randomly select img_ids
        if not isinstance(img_ids, np.ndarray):
            np.random.seed(42)
            img_ids = np.random.choice(range(len(self.imgs[split]['img'])), img_ids)
        imgs = self.imgs[split]['img'][img_ids]
        cond = None
        if is_conditional:
            cond = self.imgs[split]['cond'][img_ids]
        return (imgs, cond), img_ids

    def get_metadata(self, split, columns=['mapobject_id', 'well_name', 'cell_type', 'perturbation', 'cell_cycle']):
        mapobject_ids = self.get_mapobject_ids(split)
        wells = pd.read_csv(os.path.join(DATA_DIR, 'wells_metadata.csv'), index_col=0)
        cc = pd.read_csv(os.path.join(DATA_DIR, 'cell_cycle_classification.csv'))
        metadata = self.metadata.set_index('mapobject_id').loc[mapobject_ids]
        metadata = metadata.reset_index()
        metadata = metadata.merge(wells, left_on='well_name', right_on='well_name', how='left', suffixes=('','well_'))
        metadata = metadata.merge(cc, left_on='mapobject_id_cell', right_on='mapobject_id', how='left', suffixes=('','cc_'))
        return metadata[columns]

    # creation function
    @staticmethod
    def create(params, dataset_name, dataset_dir=os.path.join(DATA_DIR, 'datasets'), n_jobs=1):
        if params is None:
            # get params from json
            params = json.load(open(os.path.join(dataset_dir, dataset_name, 'params.json')))
        log = logging.getLogger()
        log.info('Creating train/val/test datasets with params:')
        log.info(json.dumps(params, indent=4))
        p = params

        # create mpp data (with) multiprocessing
        #res = []
        #if n_jobs == 1:
        #    for data_dir in p['data_dirs']:
        #        res.append(create_mpp_data_from_dir(data_dir, p))
        #else:
        #    with multiprocessing.Pool(n_jobs) as pool:
        #        res = pool.map(partial(create_mpp_data_from_dir, p=p), p['data_dirs'])
        #train = MPPData.concat([r[0] for r in res])
        #val = MPPData.concat([r[1] for r in res])
        #test = MPPData.concat([r[2] for r in res])

        mpp_datas = {'train': [], 'val': [], 'test': []}
        for data_dir in p['data_dirs']:
            mpp_data = MPPData.from_data_dir(data_dir, dir_type=p['dir_type'], seed=p['seed'])
            if p['mcu_dir']:
                mpp_data.add_mcu(p['mcu_dir'])
            if p['normalise']:
                mpp_data.subtract_background(p['background_value'])
            train, val, test = mpp_data.train_val_test_split(p['train_frac'], p['val_frac'])
            if p['subsample']:
                train.subsample(frac=p['frac'], frac_per_obj=p['frac_per_obj'], num=p['num'], num_per_obj=p['num_per_obj'],
                                add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'])
            elif p['neighborhood']:
                train.add_neighborhood(p['neighborhood_size'])
            mpp_datas['train'].append(train)
            mpp_datas['test'].append(test)
            mpp_datas['val'].append(val)
        # merge all datasets
        train = MPPData.concat(mpp_datas['train'])
        val = MPPData.concat(mpp_datas['val'])
        test = MPPData.concat(mpp_datas['test'])
        if p['normalise']:
            rescale_values = train.rescale_intensities_per_channel(percentile=p['percentile'])
            _ = val.rescale_intensities_per_channel(rescale_values=rescale_values)
            _ = test.rescale_intensities_per_channel(rescale_values=rescale_values)
            p['normalise_rescale_values'] = list(rescale_values)
        # add conditions
        if p.get('condition', None) is not None:
            train.add_conditions(p['condition'], cell_cycle_file=p.get('cell_cycle_file', None))
            val.add_conditions(p['condition'], cell_cycle_file=p.get('cell_cycle_file', None))
            test.add_conditions(p['condition'], cell_cycle_file=p.get('cell_cycle_file', None))
        # subset to valid mapobject_ids
        if p.get('subset_to_cell_cycle', False):
            train.subset(cell_cycle_file=p['cell_cycle_file'])
            val.subset(cell_cycle_file=p['cell_cycle_file'])
            test.subset(cell_cycle_file=p['cell_cycle_file'])
        # subset to channels
        if p.get('channels', None) is not None:
            train.subset_channels(p['channels'])
            val.subset_channels(p['channels'])
            test.subset_channels(p['channels'])

        # get images for test and val dataset
        test_imgs = {'img':[], 'mcu':[], 'cond': []}
        val_imgs = {'img': [], 'mcu': [], 'cond': []}
        test_imgs['img'] = np.array(test.get_object_imgs(data='MPP', img_size=p['test_img_size']))
        val_imgs['img'] = np.array(val.get_object_imgs(data='MPP', img_size=p['test_img_size']))
        if p['mcu_dir']:
            test_imgs['mcu'] = np.array(test.get_object_imgs(data='MCU', img_size=p['test_img_size']))
            val_imgs['mcu'] = np.array(val.get_object_imgs(data='MCU', img_size=p['test_img_size']))
        if p.get('condition', None) is not None:
            test_imgs['cond'] = np.array(test.get_object_imgs(data='condition', img_size=p['test_img_size']))
            val_imgs['cond'] = np.array(val.get_object_imgs(data='condition', img_size=p['test_img_size']))
        # subsample and add neighbors to val and test
        if p['subsample']:
            val.subsample(frac=p['frac'], frac_per_obj=p['frac_per_obj'], num=p['num'], num_per_obj=p['num_per_obj'],
                                add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'])
            test.subsample(frac=p['frac'], frac_per_obj=p['frac_per_obj'], num=p['num'], num_per_obj=p['num_per_obj'],
                                add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'])
        elif p['neighborhood']:
            test.add_neighborhood(p['neighborhood_size'])
            val.add_neighborhood(p['neighborhood_size'])

        log.info('-------------------')
        log.info('created datasets:')
        log.info('train: {}'.format(str(train)))
        log.info('val: {}'.format(str(val)))
        log.info('test: {}'.format(str(test)))
        log.info('-------------------')

        # save data
        outdir = os.path.join(dataset_dir, dataset_name)
        os.makedirs(outdir, exist_ok=True)

        # save params
        json.dump(params, open(os.path.join(outdir, 'params.json'), 'w'), indent=4)
        # save datasets
        np.savez(os.path.join(outdir, 'train_dataset.npz'), x=train.mpp, y=train.center_mpp,
                 c=train.conditions, mapobject_id=train.mapobject_ids)
        np.savez(os.path.join(outdir, 'val_dataset.npz'), x=val.mpp, y=val.center_mpp,
                 c=val.conditions, mapobject_id=val.mapobject_ids)
        np.savez(os.path.join(outdir, 'test_dataset.npz'), x=test.mpp, y=test.center_mpp,
                 c=test.conditions, mapobject_id=test.mapobject_ids)
        # save images
        np.savez(os.path.join(outdir, 'test_imgs.npz'), mapobject_id=test.metadata.mapobject_id, **test_imgs)
        np.savez(os.path.join(outdir, 'val_imgs.npz'), mapobject_id=val.metadata.mapobject_id, **val_imgs)

        # save metadata
        pd.concat([train.metadata, val.metadata, test.metadata]).to_csv(os.path.join(outdir, 'metadata.csv'))
        train.channels.to_csv(os.path.join(outdir, 'channels.csv'))


class MPPData:
    def __init__(self, metadata, channels, labels, x, y, mpp, mapobject_ids, mcu_ids=None, conditions=None, seed=42):
        self.log = logging.getLogger(self.__class__.__name__)
        np.random.seed(seed)
        self.mapobject_ids = mapobject_ids
        self.log.info('creating new MPPData with {}'.format(self.mapobject_ids))
        # subset metadata to mapobbject_ids
        self.metadata = metadata[metadata.mapobject_id.isin(np.unique(self.mapobject_ids))]
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

    def filter_cells(self, filter_criteria=['is_border_cell', 'is_mitotic'], filter_values=[0, 0]):
        """
        Filter cells given the desired criteria (from metadata).
        Input:
            -filter_criteria: list containing the metadata column
             names
            -filter_values: list of desired values corresponding to
             filter_criteria entrances
        Output:
            modify self atributes, so metadata lables, x, y,
            mapobject_ids, mpp, conditions and mcu_ids only have cell
            information that fulfill the given parameters
        """

        if len(filter_criteria) != len(filter_values):
            raise Exception('length of filter_criteria and filter_values defined in input parameters does not match!')

        metadata_mask = np.ones(self.metadata.shape[0]).astype(bool)
        print('Total number of cells: {}\n'.format(int(np.sum(metadata_mask))))
        for f, f_val in zip(filter_criteria, filter_values):
            mask_temp = (self.metadata[f] == f_val).values
            print('{} cells cutted by filter: {} == {}'.format(self.metadata.shape[0]-np.sum(mask_temp), f, f_val))
            metadata_mask &= mask_temp
        print('\nFinal number of cells cutted: {}'.format(int(self.metadata.shape[0] - np.sum(metadata_mask))))

        # Filter metadata
        self.metadata = self.metadata.iloc[metadata_mask]

        # Get mapobject_ids from metadata that fulfill the given conditions
        mapobject_ids = self.metadata.mapobject_id.values
        # Get mask and filter lables, x, y, mapobject_ids, mpp, conditions and mcu_ids
        mask = np.in1d(self.mapobject_ids, mapobject_ids)

        self.labels = self.labels[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.mapobject_ids = self.mapobject_ids[mask]
        self.mpp = self.mpp[mask]
        self.mcu_ids = self.mcu_ids[mask]
        self.conditions = self.conditions[mask]

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
            #ind = np.concatenate(ind, axis=0)f
            ind = np.in1d(self.mapobject_ids, split_ids)
            splits.append(MPPData(metadata=self.metadata, channels=self.channels, labels=self.labels[ind],
                                  x=self.x[ind], y=self.y[ind], mpp=self.mpp[ind],
                                  mapobject_ids=self.mapobject_ids[ind], mcu_ids=self.mcu_ids[ind],
                                  conditions=self.conditions[ind]))
        return splits

    #---- Functions for modify the MPPData in place ----
    def add_mcu(self, mcu_dir):
        """
        Add mcu information to the object. Should be done before any subsampling
        """
        som_to_mcu_ids = pd.read_csv(os.path.join(mcu_dir, 'leiden_som_to_mcu_ids.csv')).set_index('som_cluster_id')
        mapobject_ids = np.load(os.path.join(mcu_dir, 'mapobject_ids.npy'))
        x = np.load(os.path.join(mcu_dir, 'x.npy'))
        y = np.load(os.path.join(mcu_dir, 'y.npy'))
        cluster_ids = np.load(os.path.join(mcu_dir, 'cluster_ids.npy'))

        # iterate over all mapobject ids and find MCU cluster id
        self.log.info("Adding MCU cluster ids from {}".format(mcu_dir))
        for mapobject_id in np.unique(self.mapobject_ids):
            mask = self.mapobject_ids == mapobject_id
            mcu_mask = mapobject_ids == mapobject_id
            assert (self.x[mask] == x[mcu_mask]).all(), "x is not in correct order for matching mcu to mpp"
            assert (self.y[mask] == y[mcu_mask]).all(), "y is not in correct order for matching mcu to mpp"
            self.mcu_ids[mask] = np.array(som_to_mcu_ids.loc[cluster_ids[mcu_mask]].mcu_id)

    def _get_per_mpp_value(self, per_cell_value):
        """takes list of values corresponding to self.metadata.mapobject_id
        and propagates them to self.mapobject_id"""
        per_cell_df = pd.DataFrame({'val': per_cell_value, 'mapobject_id': self.metadata.mapobject_id})
        df = pd.DataFrame({'mapobject_id': self.mapobject_ids})
        per_mpp_value = df.merge(per_cell_df, left_on='mapobject_id', right_on='mapobject_id', how='left')['val']
        return per_mpp_value

    def add_conditions(self, cond_desc, cell_cycle_file=None):
        """
        Add conditions informations by aggregating over channels (per cell) or reading data from cell cycle file.
        Implemented conditions: '00_EU' and 'cell_cycle'
        """
        conditions = []
        for desc in cond_desc:
            cond = None
            if 'cell_cycle' in desc:
                # read cell cycle file and match with mapobjectids
                cell_cycle = pd.read_csv(cell_cycle_file)
                cond = np.array(self.metadata.merge(cell_cycle, left_on='mapobject_id_cell', right_on='mapobject_id', how='left', suffixes=('','_cc'))['cell_cycle'])
                cond = self._get_per_mpp_value(cond)
                if desc == 'cell_cycle_one_hot':
                    cond = convert_cell_cycle(np.array(cond), one_hot=True)
                else:
                    cond = convert_cell_cycle(np.array(cond)[:,np.newaxis])
            elif 'perturbation' in desc:
                # read well info file and match with mapobjectids
                wells = pd.read_csv(os.path.join(DATA_DIR, 'wells_metadata.csv'))
                cond = np.array(self.metadata.merge(wells, left_on='well_name', right_on='name', how='left')['perturbation'])
                cond = self._get_per_mpp_value(cond)
                if desc =='perturbation_one_hot':
                    cond = convert_perturbations(np.array(cond), one_hot=True)
                else:
                    cond = convert_perturbations(np.array(cond))[:, np.newaxis]

            elif desc in list(self.channels.name):
                # desc is channel - get values for this channel
                cid = list(self.channels.name).index(desc)
                cond = self.center_mpp[:,cid]
                # calculate mean channel value per cell
                df = pd.DataFrame({'cond': cond, 'mapobject_ids': self.mapobject_ids})
                df_mean = df.groupby('mapobject_ids').mean()
                cond = np.array(df.merge(df_mean, right_index=True, left_on='mapobject_ids', how='left')['cond_y'])[:,np.newaxis]
            else:
                raise NotImplementedError
            conditions.append(cond)
        self.conditions = np.concatenate(conditions, axis=-1)

    def add_cell_cycle_to_metadata(self, cc_file):
        """
        Add Cell cycle information to metadata.
        Input: Absolute path to cell cycle file
        Output: self.metadata with cell cycle information
        """

        if not os.path.exists(cc_file):
            raise Exception('Cell cycle file {} not found!'.format(cc_file))

        cc_data = pd.read_csv(cc_file)
        self.metadata = self.metadata.merge(cc_data,
                                    left_on='mapobject_id_cell',
                                    right_on='mapobject_id',
                                    how='left',
                                    suffixes=('','_cc'))
        self.metadata = self.metadata.drop(['mapobject_id_cc'], axis=1)


    def subset(self, cell_cycle_file=None):
        """\
        Subset objects to mapobject_ids listed in fname. Use
        """
        mask = np.ones(len(self.mpp), dtype=np.bool)
        if cell_cycle_file is not None:
            cell_cycle = pd.read_csv(cell_cycle_file)
            cond = np.array(self.metadata.merge(cell_cycle, left_on='mapobject_id_cell', right_on='mapobject_id', how='left', suffixes=('','_cc'))['cell_cycle'])
            cond = self._get_per_mpp_value(cond)
            mask &= ~cond.isnull()
            self.log.info(f'Subsetting mapobject ids to cell_cycle_file (removing {sum(cond.isnull())} objects)')

        else:
            self.log.warn('Called subset, but cell_cycle_file is none')

        # apply mask to data in self
        self.labels = self.labels[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.mpp = self.mpp[mask]
        self.mcu_ids = self.mcu_ids[mask]
        self.mapobject_ids = self.mapobject_ids[mask]
        self.metadata = self.metadata[self.metadata.mapobject_id.isin(np.unique(self.mapobject_ids))]
        self.conditions = self.conditions[mask]


    def subset_channels(self, channels):
        """
        Restrict self.mpp to defined channels. Channels are given as string values.
        Updates self.mpp and self.channels
        """
        cids = list(self.channels.set_index('name').loc[channels]['channel_id'])
        self.channels = self.channels.loc[cids]
        self.mpp = self.mpp[:,:,:,cids]
        self.log.info('restricted channels to {} channels'.format(len(self.channels)))

    def subsample(self, frac=None, frac_per_obj=None, num=None, num_per_obj=None, add_neighborhood=False, neighborhood_size=3):
        """
        Subsample MPPData based on selecting mpps.
        All other information is updated accordingly (to save RAM/HDD-memory)
        """
        assert sum([frac!=None, frac_per_obj!=None, num!=None, num_per_obj!=None]) == 1, "set only one of the params to a value"
        assert not (self.has_neighbor_data and add_neighborhood), "cannot add neighborhood, already has neighbor data"
        if frac is not None:
            num = int(len(self.mpp)*frac)
        if num is not None:
            self.log.info('subsampling data to {} (from {})'.format(num, len(self.mpp)))
            # randomly select num mpps
            # NOTE: need to seed default_rng, np.random.seed does not work
            rng = np.random.default_rng(seed=self.seed)
            selected = rng.choice(len(self.mpp), num, replace=False)
            # select other information accordingly
            labels = self.labels[selected]
            x = self.x[selected]
            y = self.y[selected]
            mpp = self.mpp[selected]
            mapobject_ids = self.mapobject_ids[selected]
            mcu_ids = self.mcu_ids[selected]
            conditions = self.conditions[selected]
        if frac_per_obj is not None or num_per_obj is not None:
            self.log.info('subsampling data with frac_per_obj/num_per_obj')
            # iterate over all object_ids
            labels = []
            x = []
            y = []
            mpp = []
            mapobject_ids = []
            mcu_ids = []
            conditions = []
            for mapobject_id in np.unique(self.mapobject_ids):
                obj_mask = self.mapobject_ids == mapobject_id
                if frac_per_obj is not None:
                    cur_num = int(obj_mask.sum() * frac_per_obj)
                else:
                    cur_num = num_per_obj
                rng = np.random.default_rng(seed=self.seed)
                selected = rng.choice(obj_mask.sum(), cur_num, replace=False)
                # select
                labels.append(self.labels[obj_mask][selected])
                x.append(self.x[obj_mask][selected])
                y.append(self.y[obj_mask][selected])
                mpp.append(self.mpp[obj_mask][selected])
                mapobject_ids.append(self.mapobject_ids[obj_mask][selected])
                mcu_ids.append(self.mcu_ids[obj_mask][selected])
                conditions.append(self.conditions[obj_mask][selected])
            labels = np.concatenate(labels, axis=0)
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            mpp = np.concatenate(mpp, axis=0)
            mapobject_ids = np.concatenate(mapobject_ids, axis=0)
            mcu_ids = np.concatenate(mcu_ids, axis=0)
            conditions = np.concatenate(conditions, axis=0)
        if add_neighborhood:
            self.log.info('adding neighborhood of size {}'.format(neighborhood_size))
            neighbor_mpp = self.get_neighborhood(mapobject_ids, x, y, size=neighborhood_size)
            assert (neighbor_mpp[:,neighborhood_size//2,neighborhood_size//2,:] == mpp[:,0,0,:]).all()
            mpp = neighbor_mpp
        # update self
        self.labels=labels
        self.x=x
        self.y=y
        self.mpp=mpp
        self.mapobject_ids=mapobject_ids
        self.mcu_ids=mcu_ids
        self.conditions=conditions
        self.metadata = self.metadata[self.metadata.mapobject_id.isin(np.unique(self.mapobject_ids))]

    def add_neighborhood(self, size=3):
        assert not self.has_neighbor_data, "cannot add neighborhood, already has neighbor data"
        self.log.info('adding neighborhood of size {}'.format(size))
        mpp = self.get_neighborhood(self.mapobject_ids, self.x, self.y, size=size)
        assert (mpp[:,size//2,size//2,:] == self.center_mpp).all()
        self.mpp = mpp

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
        # Note mpp is modified in place and function returns None
        if rescale_values is None:
            rescale_values = np.percentile(self.center_mpp, percentile, axis=0)
        self.log.info('rescaling mpp intensities per channel with values {}'.format(rescale_values))
        self.mpp /= rescale_values
        self.log.info('converting mpp values to float32')
        self.mpp = self.mpp.astype(np.float32)
        return rescale_values

    # ---- getting functions -----
    def get_neighborhood(self, mapobject_ids, xs, ys, size=3):
        """return neighborhood information for given mapobject_ids + xs + ys"""
        data = np.zeros((len(mapobject_ids), size, size, len(self.channels)), dtype=self.mpp.dtype)
        for mapobject_id in np.unique(mapobject_ids):
            mask = mapobject_ids == mapobject_id
            img, (xoff, yoff) = self.get_mpp_img(mapobject_id, pad=size//2)
            vals = []
            for x, y in zip(xs[mask], ys[mask]):
                idx = tuple(slice(pp-size//2, pp+size//2+1) for pp in [y-yoff,x-xoff])
                vals.append(img[idx])
            data[mask] = np.array(vals)
        return data

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

    def get_mcu_img(self, mapobject_id, **kwargs):
        """
        Calculate MCU image of given mapobject
        kwargs: arguments for get_img_from_data
        """
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = self.mcu_ids[mask][:,np.newaxis]
        return self.get_img_from_data(x, y, data, **kwargs)

    def get_mpp_img(self, mapobject_id, channel_ids=None, **kwargs):
        """
        Calculate MPP image of given mapobject
        channel_ids: ids of MPP channels that the image should hav. If None, all channels are returned.
        kwargs: arguments for get_img_from_data
        """
        if channel_ids is None:
            channel_ids = range(len(self.channels))
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = self.center_mpp[mask][:,channel_ids]
        return self.get_img_from_data(x, y, data, **kwargs)

    def get_condition_img(self, mapobject_id, **kwargs):
        """
        Calculate condition image of given mapobject
        kwargs: arguments for get_img_from_data
        """
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = self.conditions[mask]
        return self.get_img_from_data(x, y, data, **kwargs)

    def get_object_imgs(self, data='MPP', channel_ids=None, img_size=None, pad=0):
        imgs = []
        for mapobject_id in self.metadata.mapobject_id:
            if data == 'MPP':
                res = self.get_mpp_img(mapobject_id, channel_ids, img_size=img_size, pad=pad)
            elif data == 'MCU':
                res = self.get_mcu_img(mapobject_id, img_size=img_size, pad=pad)
            elif data == 'condition':
                res = self.get_condition_img(mapobject_id, img_size=img_size, pad=pad)
            else:
                raise NotImplementedError
            if img_size is None:
                res = res[0]
            imgs.append(res)
        return imgs

# test functions for MPPData
def test_get_neighborhood():
    mpp_data = MPPData.from_data_dir(data_dir=os.path.join(DATA_DIR, '184A1_UNPERTURBED/I09'), dir_type='scott')
    neigh = mpp_data.get_neighborhood(mpp_data.mapobject_ids[:1000], mpp_data.x[:1000], mpp_data.y[:1000])
    assert (mpp_data.center_mpp[:1000] == neigh[:,1,1]).all()
    return True
