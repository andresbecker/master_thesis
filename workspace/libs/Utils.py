import sys, os
from datetime import datetime
import shutil
import tensorflow as tf
import numpy as np
import logging

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

class lr_schedule_Callback(tf.keras.callbacks.Callback):
    def __init__(self, schedule, LR_SCHEDULE):
        super(lr_schedule_Callback, self).__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Learning rate scheduler class initialed')

        self.schedule = schedule
        # the warmup finish must be executed only once:
        self.warmup_finished = False
        # Set criteria to finish warmup
        self.LR_SCHEDULE = LR_SCHEDULE
        self.train_MAE = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.train_MAE = logs['mean_absolute_error']

    def on_epoch_begin(self, epoch, logs=None):

        # Unfreez pretrained layers
        if not self.warmup_finished:
            if (epoch == self.LR_SCHEDULE[0][0]) or (self.train_MAE <= self.LR_SCHEDULE[0][1]):
                # Make all layers trainable
                msg = 'Unfreezing pretrained model layers...'
                self.log.info('\n\n'+msg+'\n\n')
                print(msg)
                for layer in self.model.layers:
                    #tf.keras.backend.set_value(layer.trainable, 1)
                    layer.trainable = True
                self.warmup_finished = True

        # Update lerning rate
        # Get current lr
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.

        scheduled_lr = self.schedule(epoch, lr, self.train_MAE, self.LR_SCHEDULE)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
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
