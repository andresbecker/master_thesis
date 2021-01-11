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

class Tee_Logger(object):
    """
    Duplicates sys.stdout to a log file
    To use it in a Jupyter notebook, in the cell you want to duplicate the output call:
    TeeLog = Tee_Logger(log_file_path)
    command_you_want_to_duplicate_stdoutput

    Then in the NEXT cell call:
    TeeLog.close()

    source: https://stackoverflow.com/q/616645
    """
    def __init__(self, filename="model.log", mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None

class early_stop_Callback(tf.keras.callbacks.Callback):
    def __init__(self, target_name, target_value):
        super(early_stop_Callback, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('\nEarly stop callback class initialed')

        self.target_name = target_name
        self.target_value = target_value

    def on_epoch_end(self, epoch, logs=None):
        "Stop training if target value is reached"
        current_value = logs[self.target_name]
        if current_value < self.target_value:
            msg = '\nEarly stopping triggered!'
            msg += '\nTarget name: {}'.format(self.target_name)
            msg += '\nCurrent value = {} < {} = Target value'.format(current_value, self.target_value)
            self.log.info(msg)
            print(msg)

            self.model.stop_training = True

class save_best_model_Callback(tf.keras.callbacks.Callback):
    """
    Save model checkpoint (only weights)
    """
    #https://www.tensorflow.org/guide/keras/custom_callback
    def __init__(self, monitor='val_loss', avg_sizes=[], path=None):
        super(save_best_model_Callback, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Save_best_model callback class initialed')

        self.monitor = monitor
        self.avg_sizes = avg_sizes
        self.path = path

        self.history = []
        self.avg_history = {}
        self.best_values = {}
        self.best_values['all_epochs'] = np.Inf
        for avg_size in self.avg_sizes:
            self.avg_history[str(avg_size)] = []
            self.best_values[str(avg_size)] = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        self.history.append(logs[self.monitor])

        best_val = np.min(self.history)
        if best_val < self.best_values['all_epochs']:
            msg = '\nBest value for {} found on epoch {}:\n{}'.format(self.monitor, epoch, round(best_val,4))
            self.log.info(msg)
            print(msg)

            self.model.save_weights(self.path+'/all_epochs_best/ckpt', overwrite=True)

        for avg_size in self.avg_sizes:
            avg = np.mean(self.history[-avg_size:])
            self.avg_history[str(avg_size)].append(avg)

            if avg < self.best_values[str(avg_size)]:
                msg = '\nBest value for {} and average size {} found on epoch {}:\n{}'.format(self.monitor, avg_size, epoch, round(avg,4))
                self.log.info(msg)
                print(msg)

                self.model.save_weights(self.path+'/avg_'+str(avg_size)+'_best/ckpt', overwrite=True)


class lr_schedule_Callback(tf.keras.callbacks.Callback):
    #https://www.tensorflow.org/guide/keras/custom_callback
    def __init__(self, schedule, current_lr, LR_SCHEDULE):
        super(lr_schedule_Callback, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Learning rate scheduler callback class initialed')

        self.schedule = schedule
        # Set criteria to finish warmup
        self.LR_SCHEDULE = LR_SCHEDULE
        self.train_MAE = np.Inf
        self.current_lr = current_lr

    def on_epoch_end(self, epoch, logs=None):
        self.train_MAE = logs['mean_absolute_error']

    def on_epoch_begin(self, epoch, logs=None):

        scheduled_lr = self.schedule(epoch, self.current_lr, self.train_MAE, self.LR_SCHEDULE)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        self.current_lr = scheduled_lr
        #print(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
        #print('')

def lr_schedule(epoch, current_lr, MAE, LR_SCHEDULE):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    log = logging.getLogger()

    new_lr = np.Inf
    for i in range(len(LR_SCHEDULE)):
        if (epoch == LR_SCHEDULE[i][0]) or (MAE <= LR_SCHEDULE[i][1]):
            #print(epoch, LR_SCHEDULE[i][0], MAE, LR_SCHEDULE[i][1])
            sch_epoch = LR_SCHEDULE[i][0]
            sch_MAE = LR_SCHEDULE[i][1]
            new_lr = LR_SCHEDULE[i][2]

    # avoid returning a higher lr than the current
    if current_lr > new_lr:
        msg = '\nLearning rate updated at epoch {}'.format(epoch)
        msg += '\n\tFormer lr: {}, new lr: {}'.format(current_lr, new_lr)
        msg += '\n\tSchedul epoch: {}, schedule MAE:{}, train_MAE: {}'.format(sch_epoch, sch_MAE, MAE)
        log.info(msg+'\n')
        print(msg)

        return new_lr

    return current_lr

def create_model_dirs(parameters: dict):

    # clean_model_dir not given, then set as defaul
    if not 'clean_model_dir' in parameters.keys():
        parameters['clean_model_dir'] = True

    # Check if path where model instances are saved exist
    temp_path = os.path.join(parameters['model_path'], parameters['model_name'])
    try:
        os.makedirs(temp_path, exist_ok=True)
    except OSError as e:
        msg  = 'Dir {} could not be created!\n\nOSError: {}'.format(temp_path, e)
        raise Exception(msg)

    # Create directory where this execution will be saved
    try:
        tag = parameters['model_dir']
    except:
        # tagged with the date and time
        tag = datetime.now().strftime("%d%m%y_%H%M")

    base_path = os.path.join(temp_path, tag)
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

    def __init__(self, p, model, input_ids):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('evaluate_model class initialed')

        # model parameters
        self.p = p
        self.model = model
        # List containing the ids of the channels to be used in the prediction
        self.input_ids = input_ids

        self._load_dataset()
        self._calculate_predictions()
        self._get_metrics()

    def _calculate_predictions(self):
        columns = ['y', 'y_hat', 'mapobject_id_cell', 'set']
        self.targets_df = pd.DataFrame(columns=columns)

        dss = [self.train_data, self.val_data, self.test_data]
        ds_names = ['train', 'val', 'test']
        for ds, dsn in zip(dss, ds_names):
            for cells in ds:
                cell_ids = [cell_id.decode() for cell_id in cells['mapobject_id_cell'].numpy()]
                cell_ids = np.asarray(cell_ids).reshape(-1,1)
                cell_imgs, Y = Data_augmentation.filter_channels(cells['image'], cells['target'], self.input_ids, self.metadata)
                Y = Y.numpy()
                Y_hat = self.model.predict(cell_imgs)
                temp_df = pd.DataFrame(np.concatenate((Y, Y_hat), axis=1), columns=['y', 'y_hat'])
                temp_df['mapobject_id_cell'] = cell_ids
                temp_df['set'] = dsn
                self.targets_df = pd.concat((self.targets_df, temp_df), axis=0, ignore_index=True)

        self.targets_df['y - y_hat'] = self.targets_df.y - self.targets_df.y_hat

        # Add perturbation info to the targets df
        with open(os.path.join(self.p['pp_path'], 'metadata.csv'), 'r') as file:
            row_data_metadata = pd.read_csv(file)
            row_data_metadata.mapobject_id_cell = row_data_metadata.mapobject_id_cell.astype(str)
        per_df = row_data_metadata[['mapobject_id_cell', 'perturbation']]
        self.targets_df = self.targets_df.merge(
                per_df,
                left_on='mapobject_id_cell',
                right_on='mapobject_id_cell',
                how='left',
        )

    def _load_dataset(self):

        dataset, self.metadata = tfds.load(
            name=self.p['tf_ds_name'],
            data_dir=self.p['local_tf_datasets'],
            # If False, returns a dictionary with all the features
            as_supervised=False,
            shuffle_files=False,
            with_info=True)

        self.train_data, self.val_data, self.test_data = dataset['train'], dataset['validation'], dataset['test']

        BATCH_SIZE = self.p['BATCH_SIZE']
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.train_data = self.train_data.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        self.val_data = self.val_data.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        self.test_data = self.test_data.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    def _get_metrics(self):

        huber_loss = tf.keras.losses.Huber()

        # Create df to store metrics to compare models
        metric_columns = ['model', 'set', 'R2', 'BIC', 'MSE', 'MAE', 'Huber']
        self.metrics_df = pd.DataFrame(columns=metric_columns)

        for ss in np.unique(self.targets_df['set']):
            mask = (self.targets_df['set'] == ss)
            y = self.targets_df.y[mask].values
            y_hat_vals = self.targets_df.y_hat[mask].values
            r2 = r2_score(y, y_hat_vals)
            bic = 0
            mse = mean_squared_error(y, y_hat_vals)
            mae = mean_absolute_error(y, y_hat_vals)
            huber = huber_loss(y, y_hat_vals).numpy()
            self.metrics_df = self.metrics_df.append({'model':'y_hat', 'set':ss, 'R2':r2, 'BIC':bic, 'MSE':mse, 'MAE':mae, 'Huber':huber}, ignore_index=True)

        self.metrics_df = self.metrics_df.round(4)

    def plot_error_dist(self, figsize=(20,7), hue='perturbation'):

        col_names = ['set', 'perturbation', 'y - y_hat']
        temp_df = self.targets_df[col_names].copy()
        temp_df= temp_df.set_index(['set', 'perturbation'])
        temp_df = temp_df.stack().reset_index()
        temp_df.columns = ['set', 'perturbation'] + ['diff_name', 'value']

        plt.figure(figsize=figsize)
        sns.kdeplot(x='value',
                    data=temp_df,
                    hue=hue,
                    #color=colors,
                    shade=True,
                    bw_method=0.2)

        plt.xlabel('y - y_hat')
        plt.title('Error KDE per model')

        #plt.figure(figsize=(len(y_models)*7,7))
        #sns.boxplot(y='value',
        #            x='diff_name',
        #            hue='perturbation',
        #            data=temp_df)
        #plt.xlabel('Diff Name')
        #plt.ylabel('y - y_hat')
        #plt.title('Error Distribution per set')

    def plot_y_dist(self, figsize=(15,7)):
        temp_df = self.targets_df.copy()
        columns = ['y', 'y_hat', 'mapobject_id_cell', 'set']
        temp_df = temp_df[columns]
        #temp_df = df.loc[:, df.columns != 'perturbation']
        temp_df = temp_df.set_index(['mapobject_id_cell', 'set']).stack().reset_index()
        temp_df.columns = ['mapobject_id_cell', 'set', 'var', 'value']

        plt.figure(figsize=figsize)
        sns.boxplot(y='value',
                    x='var',
                    hue='set',
                    data=temp_df)
        plt.title('Transcription Rate (TR) values distribution')

    def plot_residuals(self, figsize=(10,7), hue='perturbation'):

        min_val = int(self.targets_df['y'].min()) - 50
        max_val = int(self.targets_df['y'].max()) + 50

        temp_df = self.targets_df.copy()
        std = temp_df['y - y_hat'].std()
        temp_df['std_residuals'] = temp_df['y - y_hat'] / std

        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=temp_df,
            x = 'y',
            y = 'std_residuals',
            hue = hue,
        )
        plt.hlines(y=2, xmin=min_val, xmax=max_val, color='red', ls='dashed')
        plt.hlines(y=-2, xmin=min_val, xmax=max_val, color='red', ls='dashed')
        plt.xlim([min_val, max_val])
        plt.ylim([-7, 7])
        plt.title('(y-y_hat)/std(y-y_hat)')

    def plot_y_vs_y_hat(self, figsize=(7,7), hue='perturbation'):

        min_val = 0
        max_val = int(self.targets_df.y.max()) + 100

        plt.figure(figsize=figsize)
        plt.axis('equal')
        sns.scatterplot(data=self.targets_df,
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
        plt.title('y vs y_hat')

    def save_model_evaluation_data(self, base_path, eval_name='test'):

        path = os.path.join(base_path, eval_name)
        os.makedirs(path, exist_ok=True)
        # Sava targets info
        with open(os.path.join(path, 'targets.csv'), 'w') as file:
            self.targets_df.to_csv(file, index=False)

        # Sava metrics info
        with open(os.path.join(path, 'metrics.csv'), 'w') as file:
            self.metrics_df.to_csv(file, index=False)
