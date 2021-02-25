import sys, os
from datetime import datetime
import shutil
import tensorflow as tf
import numpy as np
import logging
import tensorflow_datasets as tfds
import pandas as pd
import Data_augmentation
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import copy
import socket
import time

def print_stdout_and_log(msg):
    log = logging.getLogger()
    log.info(msg)
    print(msg)

def set_GPU_config(disable_gpu=False, limit_mem_growth=False):
    log = logging.getLogger()

    if disable_gpu:
        # make GPU unvisible for tf
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print_stdout_and_log('Cuda devices (GPUs) disabled')

    # list GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print_stdout_and_log('Physical GPU devises:\n{}'.format(physical_devices))

    # avoid tf to fill all the GPU memory
    if limit_mem_growth:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            msg = 'GPU Memory limited!'
        except:
            msg = 'It was not possible to limit GPU memory'

        print_stdout_and_log(msg)

def create_directory(dir_path=None, clean_if_exist=False):
    log = logging.getLogger()

    if os.path.exists(dir_path) and clean_if_exist:
        msg = 'Warning! Directory {} already exist! Deleting...\n'.format(dir_path)
        log.info(msg)
        print(msg)
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            msg = 'Dir {} could not be deleted!\n\nOSError: {}'.format(outdir, e)
            log.info(msg)
            print(msg)

    msg = 'Creating dir: {}'.format(dir_path)
    log.info(msg)
    print(msg)
    os.makedirs(dir_path, exist_ok=True)

def create_model_dirs(parameters: dict):

    # Check if the path where model instances are saved exist
    temp_path = os.path.join(parameters['model_path'], parameters['model_name'])
    try:
        os.makedirs(temp_path, exist_ok=True)
    except OSError as e:
        msg  = 'Dir {} could not be created!\n\nOSError: {}'.format(temp_path, e)
        raise Exception(msg)

    # Create directory where this execution will be saved
    base_path = parameters['base_path']
    if os.path.exists(base_path) & parameters['clean_model_dir']:
        msg = 'Warning! Directory {} already exist! Deleting...\n'.format(base_path)
        print(msg)
        try:
            shutil.rmtree(base_path)
        except OSError as e:
            msg  = 'Dir {} could not be deleted!\n\nOSError: {}'.format(base_path, e)
            raise Exception(msg)
        msg = 'Creating dir: {}'.format(base_path)
        print(msg)
    os.makedirs(base_path, exist_ok=True)

    # Create dir for model and checkpoint saiving
    model_path = os.path.join(base_path, 'model')
    try:
        os.makedirs(model_path, exist_ok=True)
    except:
        msg  = 'Dir {} could not be created!'.format(model_path)
        raise Exception(msg)

    checkpoints_path = os.path.join(base_path, 'checkpoints')
    try:
        os.makedirs(checkpoints_path, exist_ok=True)
    except:
        msg  = 'Dir {} could not be created!'.format(checkpoints_path)
        raise Exception(msg)

    return base_path, model_path, checkpoints_path

class evaluate_model():

    def __init__(self, p, model, projection_tensor, metadata_df, metrics):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('evaluate_model class initialed')

        # model parameters
        self.p = p
        self.model = model
        self.metrics = metrics
        self.metadata_df = metadata_df
        self.projection_tensor = projection_tensor

        self._get_predictions()

    def _get_predictions(self):
        # First load TFDS
        self._load_dataset()

        columns = ['y', 'y_hat', 'mapobject_id_cell', 'set']
        self.targets_df = pd.DataFrame(columns=columns)

        dss = [self.train_data, self.val_data, self.test_data]
        ds_names = ['train', 'val', 'test']
        for ds, dsn in zip(dss, ds_names):
            for cells in ds:
                cell_ids = [cell_id.decode() for cell_id in cells['mapobject_id_cell'].numpy()]
                cell_ids = np.asarray(cell_ids).reshape(-1,1)
                cell_imgs, Y = Data_augmentation.apply_data_preprocessing(cells['image'], cells['target'], self.projection_tensor)
                Y = Y.numpy()
                Y_hat = self.model.predict(cell_imgs)
                temp_df = pd.DataFrame(np.concatenate((Y, Y_hat), axis=1), columns=['y', 'y_hat'])
                temp_df['mapobject_id_cell'] = cell_ids
                temp_df['set'] = dsn
                self.targets_df = pd.concat((self.targets_df, temp_df), axis=0, ignore_index=True)

        self.targets_df['y - y_hat'] = self.targets_df.y - self.targets_df.y_hat

        # Add perturbation info to the targets df
        self.metadata_df.mapobject_id_cell = self.metadata_df.mapobject_id_cell.astype(str)
        temp_df = self.metadata_df[['mapobject_id_cell', 'perturbation', 'cell_cycle']]
        self.targets_df = self.targets_df.merge(
                temp_df,
                left_on='mapobject_id_cell',
                right_on='mapobject_id_cell',
                how='left',
        )

    def _load_dataset(self, as_supervised=False):

        dataset = tfds.load(
            name=self.p['tf_ds_name'],
            data_dir=self.p['local_tf_datasets'],
            # If False, returns a dictionary with all the features
            as_supervised=as_supervised,
            shuffle_files=False,
            with_info=False)

        self.train_data, self.val_data, self.test_data = dataset['train'], dataset['validation'], dataset['test']
        del(dataset)

        BATCH_SIZE = self.p['BATCH_SIZE']
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.train_data = self.train_data.batch(BATCH_SIZE)
        self.val_data = self.val_data.batch(BATCH_SIZE)
        self.test_data = self.test_data.batch(BATCH_SIZE)

        if as_supervised:
            self.train_data = self.train_data.map(lambda image, target: Data_augmentation.apply_data_preprocessing(image, target, self.projection_tensor), num_parallel_calls=AUTOTUNE)
            self.val_data = self.val_data.map(lambda image, target: Data_augmentation.apply_data_preprocessing(image, target, self.projection_tensor), num_parallel_calls=AUTOTUNE)
            self.test_data = self.test_data.map(lambda image, target: Data_augmentation.apply_data_preprocessing(image, target, self.projection_tensor), num_parallel_calls=AUTOTUNE)

        self.train_data = self.train_data.prefetch(AUTOTUNE)
        self.val_data = self.val_data.prefetch(AUTOTUNE)
        self.test_data = self.test_data.prefetch(AUTOTUNE)

    def _get_metrics_with_DA(self):
        """
        This method evaluate the model using the model's method evaluate. this means that if the model was instantiated using the custom model class, the valuation will be done over the fixed augmentation data techniques see class CustomModel, method test_step and _predict_targets_with_augmentation.
        """
        # first load the dataset with the supervised flag on
        self._load_dataset(as_supervised=True)

        # array containing the metrics name
        temp_metrics = ['Aug_'+m for m in self.metrics]

        Aug_metrics_df = pd.DataFrame(columns=temp_metrics+['Set'])

        dss = [self.train_data, self.val_data, self.test_data]
        ds_names = ['train', 'val', 'test']
        for ds, dsn in zip(dss, ds_names):
            temp_data = self.model.evaluate(ds, verbose=0)
            temp_dict = {}
            # save metrics names and eval values together
            for met, m_eval in zip(temp_metrics, temp_data):
                temp_dict[met] = m_eval
            temp_dict['Set'] = dsn
            # add to dataframe
            Aug_metrics_df = Aug_metrics_df.append(temp_dict, ignore_index=True)

        # merge Aug_metrics_df and metrics_df
        self.metrics_df = self.metrics_df.merge(Aug_metrics_df,
                                                left_on='Set',
                                                right_on='Set',
                                                how='left'
                                                )

    def get_metrics(self, CMA_size=0, CMA=0, CMA_Std=0, Epoch=0):

        huber_loss = tf.keras.losses.Huber()

        # Create df to store metrics to compare models
        columns = ['Model', 'Loss', 'lr', 'N_Epochs', 'Conv_L1_reg', 'Conv_L2_reg', 'Dense_L1_reg', 'Dense_L2_reg', 'Bias_l2_reg', 'PreTrained', 'Aug_rand_h_flip', 'Aug_rand_90deg_r', 'Aug_Zoom', 'Aug_Zoom_mode', 'Aug_rand_int', 'Aug_RI_mean', 'Aug_RI_stddev', 'Set', 'Bias', 'Std', 'R2', 'MAE', 'MSE', 'Huber', 'CMA_size', 'CMA', 'CMA_Std', 'Epoch', 'DS_name', 'custom_model_class', 'Early_stop_patience', 'Parameters_file_path']

        self.metrics_df = pd.DataFrame(columns=columns)

        for ss in np.unique(self.targets_df['set']):
            mask = (self.targets_df['set'] == ss)
            y = self.targets_df.y[mask].values
            y_hat_vals = self.targets_df.y_hat[mask].values
            r2 = r2_score(y, y_hat_vals)
            bic = 0
            mse = mean_squared_error(y, y_hat_vals)
            mae = mean_absolute_error(y, y_hat_vals)
            huber = huber_loss(y, y_hat_vals).numpy()
            temp_dict = {'Model':self.p['model_name'],
                        'Loss':self.p['loss'],
                        'lr':self.p['learning_rate'],
                        'N_Epochs':self.p['number_of_epochs'],
                        'Conv_L1_reg':self.p['conv_reg'][0],
                        'Conv_L2_reg':self.p['conv_reg'][1],
                        'Dense_L1_reg':self.p['dense_reg'][0],
                        'Dense_L2_reg':self.p['dense_reg'][1],
                        'Bias_l2_reg':self.p['bias_l2_reg'],
                        'PreTrained':self.p['pre_training'],
                        'Aug_rand_h_flip':self.p['random_horizontal_flipping'],
                        'Aug_rand_90deg_r':self.p['random_90deg_rotations'],
                        'Aug_Zoom':self.p['CenterZoom'],
                        'Aug_Zoom_mode':self.p['CenterZoom_mode'],
                        'Aug_rand_int':self.p['Random_channel_intencity'],
                        'Aug_RI_mean':self.p['RCI_mean'],
                        'Aug_RI_stddev':self.p['RCI_stddev'],
                        'Set':ss,
                        'Bias':self.targets_df['y - y_hat'][mask].mean(),
                        'Std':self.targets_df['y - y_hat'][mask].std(),
                        'R2':r2,
                        'MAE':mae,
                        'MSE':mse,
                        'Huber':huber,
                        'CMA_size':CMA_size,
                        'CMA':CMA,
                        'CMA_Std':CMA_Std,
                        'Epoch':Epoch,
                        'DS_name': self.p['tf_ds_name'],
                        'custom_model_class': self.p['custom_model_class'],
                        'Early_stop_patience': self.p['early_stop_patience'],
                        'Parameters_file_path':self.p['parameters_file_path']}

            self.metrics_df = self.metrics_df.append(temp_dict, ignore_index=True)

        # if custom model calss selected, evaluate the dataset using the fixed data augm. techn
        if self.p['custom_model_class']:
            self._get_metrics_with_DA()

        self.metrics_df = self.metrics_df.round(4)

    def plot_error_dist(self, figsize=(20,7), sets=['train', 'val', 'test'], title=''):

        col_names = ['set', 'perturbation', 'cell_cycle', 'y - y_hat']
        temp_df = self.targets_df[col_names].copy()
        temp_df= temp_df.set_index(['set', 'perturbation', 'cell_cycle'])
        temp_df = temp_df.stack().reset_index()
        temp_df.columns = ['set', 'perturbation', 'cell_cycle'] + ['diff_name', 'value']
        hues = ['cell_cycle', 'perturbation']

        n_plots = len(sets) + len(hues)
        plt.figure(figsize=(n_plots*figsize[0], figsize[1]))
        plot_count = 0
        for hue in hues:
            for s in sets:
                plot_count += 1
                plt.subplot(1,n_plots,plot_count)
                sns.kdeplot(x='value',
                            data=temp_df[temp_df.set == s],
                            hue=hue,
                            #color=colors,
                            shade=True,
                            bw_method=0.2)

                plt.xlabel('y - y_hat')
                plt.title(title+', Error distribution, set={}, hue={}'.format(s, hue))

    def plot_y_dist(self, figsize=(15,7), sets=['train','val', 'test'], title=''):
        temp_df = self.targets_df.copy()
        columns = ['y', 'y_hat', 'mapobject_id_cell', 'set', 'cell_cycle', 'perturbation']
        temp_df = temp_df[columns]
        #temp_df = df.loc[:, df.columns != 'perturbation']
        temp_df = temp_df.set_index(['mapobject_id_cell', 'set', 'cell_cycle', 'perturbation']).stack().reset_index()
        temp_df.columns = ['mapobject_id_cell', 'set', 'cell_cycle', 'perturbation', 'var', 'value']

        xs = ['cell_cycle', 'perturbation']
        n_plots = len(sets) + len(xs)
        plt.figure(figsize=(n_plots*figsize[0], figsize[1]))

        plot_count = 0
        for x in xs:
            for s in sets:
                plot_count += 1
                plt.subplot(1,n_plots,plot_count)
                sns.boxplot(y='value',
                            x=x,
                            hue='var',
                            data=temp_df[temp_df.set == s])
                plt.title(title+', Transcription Rate (TR) distribution, set={}, x={}'.format(s, x))

    def plot_residuals(self, figsize=(10,7), sets=['train','val', 'test'], title=''):

        min_val = int(self.targets_df['y'].min()) - 50
        max_val = int(self.targets_df['y'].max()) + 50

        temp_df = self.targets_df.copy()
        std = temp_df['y - y_hat'].std()
        temp_df['std_residuals'] = temp_df['y - y_hat'] / std

        hues = ['cell_cycle', 'perturbation']
        n_plots = len(sets) + len(hues)
        plt.figure(figsize=(n_plots*figsize[0], figsize[1]))
        plot_count = 0
        for hue in hues:
            for s in sets:
                plot_count += 1
                plt.subplot(1,n_plots,plot_count)

                sns.scatterplot(
                    data=temp_df[temp_df.set == s],
                    x = 'y',
                    y = 'std_residuals',
                    hue = hue,
                )
                plt.hlines(y=2, xmin=min_val, xmax=max_val, color='red', ls='dashed')
                plt.hlines(y=-2, xmin=min_val, xmax=max_val, color='red', ls='dashed')
                plt.xlim([min_val, max_val])
                plt.ylim([-7, 7])
                plt.title(title+', (y-y_hat)/std(y-y_hat), set={}, hue={}'.format(s, hue))

    def plot_y_vs_y_hat(self, figsize=(7.6,7), sets=['train','val', 'test'], title=''):

        min_val = 0
        max_val = int(self.targets_df.y.max()) + 100

        hues = ['cell_cycle', 'perturbation']
        n_plots = len(sets) + len(hues)
        plt.figure(figsize=(n_plots*figsize[0], figsize[1]))
        plot_count = 0
        for hue in hues:
            for s in sets:
                plot_count += 1
                plt.subplot(1,n_plots,plot_count)
                plt.axis('equal')

                sns.scatterplot(data=self.targets_df[self.targets_df.set == s],
                                x='y',
                                y='y_hat',
                                hue=hue,
                                #s=15,
                                alpha=0.5)

                x_line = [min_val, max_val]
                y_line = x_line
                plt.plot(x_line, y_line, linestyle='dashed', color='red')
                plt.xlim([min_val, max_val])
                plt.ylim([min_val, max_val])
                plt.title(title+', y vs y_hat, set={}, hue={}'.format(s, hue))

    def save_model_evaluation_data(self, base_path, eval_name='test'):

        # Sava targets info
        with open(os.path.join(base_path, 'targets_'+eval_name+'.csv'), 'w') as file:
            self.targets_df.to_csv(file, index=False)

def plot_train_metrics(history=None, CMA_history=None, CMA_metric=None, metrics=None, p=None, title='', figsize=(15,23)):

    plt.figure(figsize=figsize)
    if CMA_history is not None:
        CMA_history = np.array(CMA_history)

    for i, metric in enumerate(metrics,1):

        # set limits for plots
        if metric == 'mse':
            min_val = 900
            max_val = 11000
        elif metric == 'mean_absolute_error':
            min_val = 25
            max_val = 100
        else:
            warm_stage = int(len(history['loss'])*0.20)
            min_val = np.asarray(history[metric]+history['val_'+metric]).min()
            max_val = np.asarray(history[metric][warm_stage:]+history['val_'+metric][warm_stage:]).max()

        plt.subplot(3,1,i)
        # Plot Train loss
        x = range(1, len(history[metric])+1)
        plt.plot(x, history[metric], alpha=0.5, label=metric, c='darkorange')
        # Plot Validation loss
        plt.plot(x, history['val_'+metric], alpha=0.5, label='val_'+metric, c='darkblue')
        # Plot CMA loss
        if (CMA_history is not None) and (metric == CMA_metric):
            plt.plot(CMA_history[:,0], CMA_history[:,1], label='avg val_'+metric, c='blue')

        # Plot a red point in the epoch with the best val metric
        val_min = np.asarray(history['val_'+metric]).min()
        val_min_idx = np.argmin(history['val_'+metric]) + 1
        label='Bets singel val\nEpoch={}\n{}={}'.format(val_min_idx,metric,round(val_min,2))
        plt.scatter(x=val_min_idx, y=val_min, c='brown', linewidths=4, label=label)

        # Plot a green point in the epoch with the best val metric
        if (CMA_history is not None) and (metric == CMA_metric):
            temp_idx = np.argmin(CMA_history[:,1])
            val_min_idx = int(CMA_history[temp_idx,0])
            val_min_CMA = CMA_history[temp_idx,1]
            val_min = history['val_'+metric][val_min_idx-1]

            label='Bets CMA val\nEpoch={}\nCMA={}'.format(val_min_idx,round(val_min,2))
            plt.scatter(x=val_min_idx, y=val_min_CMA, c='green', linewidths=4, label=label)
            label='{}={}'.format(metric,round(val_min,2))
            plt.scatter(x=val_min_idx, y=val_min, c='green', marker='x', linewidths=2, label=label)

        plt.grid(True)
        plt.ylim([min_val, max_val])
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.title(metric+', '+title)

def set_model_default_parameters(p_old=None):
    """
    This function check that all necessary input parameters are given, and if not, it set default values depending on the host name (execution in local machine or server)
    """
    p_new = {}
    p_new['parameters_file_path'] = p_old['parameters_file_path']
    p_new['external_libs_path'] = p_old['external_libs_path']

    hostname = socket.gethostname()
    local_hn = 'hughes-machine'

    info = '\nInput parameters:\n'

    # Model selection----------------------------------------------------------
    info += '\n  Model parameters:'
    key = 'model_name'
    if key not in p_old.keys():
        p_new[key] = 'ResNet50V2'
    else:
        p_new[key] = p_old[key]
    info += '\n    Selected model: ' + p_new[key]

    key = 'custom_model_class'
    if key not in p_old.keys():
        p_new[key] = True
    else:
        p_new[key] = p_old[key]
    info += '\n    Instantiate custom model class: '+str(p_new[key])

    key = 'pre_training'
    if key not in p_old.keys():
        p_new[key] = False
    else:
        p_new[key] = p_old[key]
    info += '\n    Pretraining: '+str(p_new[key])

    # Regularization: conv_reg=[l1_reg, l2_reg]=[1e-5, 1e-4]
    key = 'dense_reg'
    if key not in p_old.keys():
        p_new[key] = [0, 0]
    else:
        p_new[key] = p_old[key]
    info += '\n    Dense layers regularization (l1, l2): '+str(p_new[key])
    key = 'conv_reg'
    if key not in p_old.keys():
        p_new[key] = [0, 0]
    else:
        p_new[key] = p_old[key]
    info += '\n    Conv layers regularization (l1, l2): '+str(p_new[key])
    key = 'bias_l2_reg'
    if key not in p_old.keys():
        p_new[key] = 0
    else:
        p_new[key] = p_old[key]
    info += '\n    Bias L2 reg (for dense and conv layers): '+str(p_new[key])

    key = 'number_of_epochs'
    if key not in p_old.keys():
        p_new[key] = 450
    else:
        p_new[key] = p_old[key]
    info += '\n    Number of epochs: '+str(p_new[key])

    key = 'early_stop_patience'
    if key not in p_old.keys():
        p_new[key] = 100
    else:
        p_new[key] = p_old[key]
    info += '\n    Early stop patience: '+str(p_new[key])

    # Losses: mse, huber, mean_absolute_error
    key = 'loss'
    if key not in p_old.keys():
        p_new[key] = 'huber'
    else:
        p_new[key] = p_old[key]
    info += '\n    Loss function: '+str(p_new[key])

    key = 'learning_rate'
    if key not in p_old.keys():
        p_new[key] = 0.001
    else:
        p_new[key] = p_old[key]
    info += '\n    Learning rate: '+str(p_new[key])

    key = 'BATCH_SIZE'
    if key not in p_old.keys():
        p_new[key] = 64
    else:
        p_new[key] = p_old[key]
    info += '\n    Batch size: '+str(p_new[key])

    # verbose_level: 1=progress bar, 2=one line per epoch
    key = 'verbose_level'
    if key not in p_old.keys():
        p_new[key] = 2
    else:
        p_new[key] = p_old[key]
    info += '\n    Verbose level: '+str(p_new[key])

    info += '\n'
    # End of the model section--------------------------------------------------

    # Output section------------------------------------------------------------
    info += '\n  Output:'

    # where to save the models and checkpoints
    key = 'model_path'
    if key not in p_old.keys():
        if hostname == local_hn:
            p_new[key] = '/home/hhughes/Documents/Master_Thesis/Project/workspace/models'
        else:
            p_new[key] = '/storage/groups/ml01/code/andres.becker/master_thesis/workspace/models'
    else:
        p_new[key] = p_old[key]

    # use parameters file name as base name for other files and dirs
    key = 'basename'
    p_new[key] = os.path.basename(p_new['parameters_file_path']).split(".")[0]
    info += '\n    Base name for files: '+str(p_new[key])

    # Create output path to save model
    key = 'base_path'
    temp = os.path.basename(p_new['parameters_file_path']).split(".")[0]
    p_new[key] = os.path.join(p_new['model_path'], p_new['model_name'], temp)
    info += '\n    Model output: '+str(p_new[key])

    key = 'clean_model_dir'
    if key not in p_old.keys():
        p_new[key] = False
    else:
        p_new[key] = p_old[key]
    info += '\n    Clean base_path? '+str(p_new[key])

    info += '\n'
    # End of the output section-------------------------------------------------

    # Dataset section----------------------------------------------------------
    info += '\n  Dataset:'

    # Tensorflow dataset
    key = 'tf_ds_name'
    if key not in p_old.keys():
        p_new[key] = 'mpp_dataset_normal_dmso'
    else:
        p_new[key] = p_old[key]
    info += '\n     TFDS name: '+str(p_new[key])

    # Hannah is not using: 00_BG488, 00_BG568, 09_SRRM2_ILASTIK, 15_SON_ILASTIK
    key = 'input_channels'
    if key not in p_old.keys():
        p_new[key] = ['00_DAPI',
                  '07_H2B',
                  '01_CDK9_pT186',
                  '03_CDK9',
                  '05_GTF2B',
                  '07_SETD1A',
                  '08_H3K4me3',
                  '09_SRRM2',
                  '10_H3K27ac',
                  '11_KPNA2_MAX',
                  '12_RB1_pS807_S811',
                  '13_PABPN1',
                  '14_PCNA',
                  '15_SON',
                  '16_H3',
                  '17_HDAC3',
                  '19_KPNA1_MAX',
                  '20_SP100',
                  '21_NCL',
                  '01_PABPC1',
                  '02_CDK7',
                  '03_RPS6',
                  '05_Sm',
                  '07_POLR2A',
                  '09_CCNT1',
                  '10_POL2RA_pS2',
                  '11_PML',
                  '12_YAP1',
                  '13_POL2RA_pS5',
                  '15_U2SNRNPB',
                  '18_NONO',
                  '20_ALYREF',
                  '21_COIL']
    else:
        p_new[key] = p_old[key]
    info += '\n    Selected input channels: '
    info += '\n    '+str(p_new[key])

    key = 'shuffle_files'
    if key not in p_old.keys():
        p_new[key] = True
    else:
        p_new[key] = p_old[key]
    info += '\n    Shuffle TFDS elements in the first loading? '+str(p_new[key])

    key = 'local_tf_datasets'
    if key not in p_old.keys():
        if hostname == local_hn:
            p_new[key] = '/home/hhughes/Documents/Master_Thesis/Project/datasets/tensorflow_datasets'
        else:
            p_new[key] = '/storage/groups/ml01/workspace/andres.becker/datasets/tensorflow_datasets'
    else:
        p_new[key] = p_old[key]
    info += '\n    TFDS path: '+str(p_new[key])

    info += '\n'
    # End of the Dataset section------------------------------------------------

    # Data augmentation section-------------------------------------------------
    info += '\n  Data augmentation:'
    key = 'random_horizontal_flipping'
    if key not in p_old.keys():
        p_new[key] = True
    else:
        p_new[key] = p_old[key]
    info += '\n    Random horizontal flipping: '+str(p_new[key])

    key = 'random_90deg_rotations'
    if key not in p_old.keys():
        p_new[key] = True
    else:
        p_new[key] = p_old[key]
    info += '\n    Random 90deg rotations: '+str(p_new[key])

    key = 'CenterZoom'
    if key not in p_old.keys():
        p_new[key] = True
    else:
        p_new[key] = p_old[key]
    info += '\n    CenterZoom: '+str(p_new[key])

    # available: random_uniform (cell size randomly selected from uniform dist) equal (all cells with same size)
    key = 'CenterZoom_mode'
    if key not in p_old.keys():
        p_new[key] = 'random_normal'
    else:
        p_new[key] = p_old[key]
    info += '\n      Center zoom mode: '+str(p_new[key])

    key = 'Random_channel_intencity'
    if key not in p_old.keys():
        p_new[key] = True
    else:
        p_new[key] = p_old[key]
    info += '\n    Random_channel_intencity: '+str(p_new[key])
    key = 'RCI_mean'
    if key not in p_old.keys():
        p_new[key] = 0
    else:
        p_new[key] = p_old[key]
    info += '\n      RCH dist mean: ' + str(p_new[key])
    key = 'RCI_stddev'
    if key not in p_old.keys():
        p_new[key] = 0.1667
    else:
        p_new[key] = p_old[key]
    info += '\n      RCH dist stddev: ' + str(p_new[key])

    info += '\n'
    # End of the Data augmentation section--------------------------------------


    # Logging section-----------------------------------------------------------
    info += '\n  Logging:'
    key = 'log_path'
    if key not in p_old.keys():
        p_new[key] = '/tmp'
    else:
        p_new[key] = p_old[key]

    key = 'log_file'
    p_new[key] = os.path.join(p_new['log_path'], p_new['basename']+'.log')
    info += '\n    Log file: '+str(p_new[key])

    key = 'tensorboard'
    if key not in p_old.keys():
        p_new[key] = False
    else:
        p_new[key] = p_old[key]
    info += '\n    Print tensorboard logs? '+str(p_new[key])

    info += '\n'
    # End of the Logging section------------------------------------------------

    # other section-------------------------------------------------------------
    info += '\n  Reproducibility:'
    # seed
    key = 'seed'
    if key not in p_old.keys():
        p_new[key] = 123
    else:
        p_new[key] = p_old[key]
    info += '\n    Random Seed: '+str(p_new[key])

    info += '\n  Cuda:'
    # Cuda Config
    key = 'disable_gpu'
    if key not in p_old.keys():
        if hostname == local_hn:
            p_new[key] = True
        else:
            p_new[key] = False
    else:
        p_new[key] = p_old[key]
    info += '\n    Ignore GPU? '+str(p_new[key])

    key = 'set_memory_growth'
    if key not in p_old.keys():
        if hostname == local_hn:
            p_new[key] = True
        else:
            p_new[key] = False
    else:
        p_new[key] = p_old[key]
    info += '\n    Limit GPU memory allocation? '+str(p_new[key])

    info += '\n\n'

    return p_new, info
