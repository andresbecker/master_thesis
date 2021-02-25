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

class save_best_model_weights_in_memory_Callback(tf.keras.callbacks.Callback):
    """
    Save model checkpoint (only weights)
    """
    # https://www.tensorflow.org/guide/keras/custom_callback
    def __init__(self, monitor='val_loss'):
        super(save_best_model_weights_in_memory_Callback, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Save_best_model callback class initialed')

        self.monitor = monitor
        self.best_value = np.Inf
        self.best_model_w_and_b = []

    def on_epoch_end(self, epoch, logs=None):
        # Epoch start on 0!
        epoch += 1

        if logs[self.monitor] < self.best_value:

            msg += '\nBest general model updated on epoch {}; New value = {} < {} = last value'.format(epoch, round(self.best_value, 2), round(logs[self.monitor], 2))

            # Update value
            self.best_value = logs[self.monitor]
            # save weghts and biases in ram memory
            self.best_model_w_and_b = copy.deepcopy(self.model.get_weights())

            self.log.info(msg)
            print(msg)

class print_progress_to_log(tf.keras.callbacks.Callback):
    """
    This class (tf callback) is ment to print the training progress into a log fail in case the training is made in a Jupyter notebook.
    """
    # https://www.tensorflow.org/guide/keras/custom_callback
    def __init__(self, N_Epochs):
        super(print_progress_to_log, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('print_progress_to_log class initialed')

        self.N_Epochs = N_Epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Epoch start on 0!
        msg = '\nEpoch {}/{}'.format(epoch+1, self.N_Epochs)
        self.log.info(msg)
        self.epoch_starting_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Epoch start on 0!
        epoch += 1
        # Print metrics
        msg = '\n{}s'.format(round(time.time()-self.epoch_starting_time, 0))
        for key in logs.keys():
            msg += ' - {}: {}'.format(key, round(logs[key],2))
        self.log.info(msg)

class save_best_model_base_on_CMA_Callback(tf.keras.callbacks.Callback):
    """
    Save model checkpoint (only weights) accordingly to the Central Moving Average CMA
    """
    #https://www.tensorflow.org/guide/keras/custom_callback
    def __init__(self, monitor='val_loss', avg_sizes=[11, 21, 31], early_stop=False, patience=50):
        super(save_best_model_base_on_CMA_Callback, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('\nSave_best_model callback class initialed')

        # CMA_0 (i.e. best model without moving avg) is saved by default, but it use other means to save it, therefore remove it from avg_sizes array if it is given
        avg_sizes = [s for s in avg_sizes if s > 0]

        for s in avg_sizes:
            if (s < 3) or (s % 2 != 1):
                msg = '\nNumbers on avg_sizes most be odd numbers and bigger than 3'
                self.log.error(msg)
                raise Exception(msg)

        self.early_stop = early_stop
        self.patience = patience
        self.stopped_epoch = 0

        self.monitor = monitor
        self.history = []
        self.avg_sizes = avg_sizes
        self.avg_lags = (np.array(self.avg_sizes) + 1) // 2
        self.max_avg_lag = np.max(self.avg_lags)
        self.min_avg_lag = np.min(self.avg_lags)
        # Dictionary to save the CMA history
        self.CMA_history = {}
        # Dictionary to save the last self.max_avg_lag models
        self.model_stack = {}
        # Dictionary to save best models (depending to the CMA)
        self.best_models = {}
        for avg_size in self.avg_sizes:
            # Each self.best_models entry contains:
            # [CMA_epoch, CMA_value, CMA_std, model_weights]
            self.best_models['CMA_'+str(avg_size)] = [None, np.Inf, 0, None]
            self.CMA_history['CMA_'+str(avg_size)] = []

        self.best_models['CMA_0'] = [None, np.Inf, 0, None]
        self.CMA_history['CMA_0'] = None

    def on_epoch_end(self, epoch, logs=None):
        # Clean callback message
        msg = ''
        # Epoch start on 0!
        epoch += 1

        # Save the last max_avg_lag models
        if epoch >= self.min_avg_lag:
            self.model_stack[epoch] = copy.deepcopy(self.model.get_weights())
        # Delet last model (which is out of the CMA window)
        if epoch >= (self.min_avg_lag + self.max_avg_lag):
            del(self.model_stack[epoch-self.max_avg_lag])

        self.history.append(logs[self.monitor])

        for avg_size, avg_lag in zip(self.avg_sizes, self.avg_lags):

            if epoch >= avg_size:
                # epoch corresponding with the CMA
                CMA_epoch = epoch - avg_lag + 1
                avg = np.mean(self.history[-avg_size:])
                std = np.std(self.history[-avg_size:])

                # Save CMA history
                self.CMA_history['CMA_'+str(avg_size)].append([CMA_epoch, avg])

                if avg < self.best_models['CMA_'+str(avg_size)][1]:

                    msg += '\nBest CMA_{} model updated for epoch {}; New CMA_{} value = {} < {} = last CMA_{} value'.format(avg_size, CMA_epoch, avg_size, round(avg, 2), round(self.best_models['CMA_'+str(avg_size)][1], 2), avg_size)

                    # Update value
                    self.best_models['CMA_'+str(avg_size)] = [CMA_epoch, avg, std, self.model_stack[CMA_epoch]]

        # Save best value without considering the CMA (i.e. regular best model
        # saving)
        best_val = np.min(self.history)
        if best_val < self.best_models['CMA_0'][1]:

            msg += '\nBest general model updated for epoch {}; New value = {} < {} = last value'.format(epoch, round(best_val, 2), round(self.best_models['CMA_0'][1], 2))

            # Update values
            if epoch in self.model_stack.keys():
                self.best_models['CMA_0'] = [epoch, best_val, 0,  self.model_stack[epoch]]
            else:
                self.best_models['CMA_0'] = [epoch, best_val, 0,  copy.deepcopy(self.model.get_weights())]

        if msg != '':
            self.log.info(msg)
            print(msg)

        # Eraly stop
        if self.early_stop:
            self._triger_early_stop(epoch)

    def _triger_early_stop(self, epoch):
        """
        Check if the patience value have being exceeded for all CMA's
        """
        # if there is a moving average that have not exceeded the patience, then early_stop_triger will be turned into False
        early_stop_triger = True
        for key in self.best_models.keys():
            last_epoch = self.best_models[key][0]
            if last_epoch is not None:
                # is patience exceeded for this moving avg?
                early_stop_triger &= ((epoch - last_epoch) >= self.patience)

        if early_stop_triger:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % self.stopped_epoch)



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

def set_tensorboard(log_path='/tmp', log_dir_name='test'):
    """
    This function do all the needed stuff to set tensorboard.
    """

    log = logging.getLogger()

    tb_dir_path = os.path.join(log_path, log_dir_name+'_tensorboard')
    if os.path.exists(tb_dir_path):
        try:
            shutil.rmtree(tb_dir_path)
        except OSError as e:
            msg  = 'Tensorboard log dir {} could not be deleted!\n\nOSError: {}'.format(tb_dir_path, e)
            print(msg)
            log.info(msg)

    msg = 'Tensorboard file: {}'.format(tb_dir_path)
    print(msg)
    log.info(msg)

    return tf.keras.callbacks.TensorBoard(log_dir=tb_dir_path, histogram_freq=1)
