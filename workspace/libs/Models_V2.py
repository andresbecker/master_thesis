import tensorflow as tf
from tensorflow.image import rot90 as img_rot90
from tensorflow.image import flip_left_right as img_flip
from tensorflow.keras.optimizers import Adam
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import copy
import os
import time
#from tensorflow.python.keras.engine import data_adapter

class CustomModel(tf.keras.Model):
    """
    The logic for one evaluation step.
    This sub-calss of tf.keras.Model is ment to apply fixed augmentation techniques to the validation set, so the val dataset will be a better representation of the cell images population. This is done by overwritting the Model method test_step. More info on:
    - https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#providing_your_own_evaluation_step

    This function contains the mathemetical logic for one step of evaluation.
    This includes the forward pass, loss calculation, and metrics updates.

    Arguments:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned.
    """
    def _predict_targets_with_augmentation(self, x, y):
        """
        This function is ment to evaluate (predict) the validation test using fixed data augmentation (DA) techniques. Currently the val_dataset is evaluated 8 times (no DA, 90, 180, 270, deg rotations, flip, flip+90, flip+180, flip+270 deg rotations).
        """
        # Compute predictions without augmentation
        y_pred = self(x, training=False)

        n_repeats = 1
        # Compute predictions over imgs with k*90 deg (k=1,2,3) rotations
        for k in range(1,4):
            n_repeats += 1
            temp_pred = self(img_rot90(x, k=k), training=False)
            y_pred = tf.concat((y_pred, temp_pred), axis=0)

        # Compute predictions over imgs with k*90 deg (k=0,1,2,3) rotations+flip
        for k in range(0,4):
            n_repeats += 1
            temp_pred = self(img_rot90(img_flip(x), k=k), training=False)
            y_pred = tf.concat((y_pred, temp_pred), axis=0)

        # stack true values as many times as needed
        y = tf.repeat(y, repeats=n_repeats, axis=0)

        return y, y_pred

    def test_step(self, data):
        #data = data_adapter.expand_1d(data)
        # Unpack the data
        #x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        x, y = data

        # predict targets with and without data augmentation
        y, y_pred = self._predict_targets_with_augmentation(x, y)

        # Updates the metrics tracking the loss
        #self.compiled_loss(
        #    y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the metrics.
        #self.compiled_metrics.update_state(y, y_pred, sample_weight)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

class Predef_models():
    """
    This class contains different models
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Predef_models_and_utils class initialed')

    def select_model(self, model_name=None, input_shape=None, pre_training=False, conv_reg=[0,0], dense_reg=[0,0], bias_l2_reg=0, return_custom_model=True):

        self.model_name = model_name
        self.pre_training = pre_training
        self.return_custom_model = return_custom_model

        if input_shape is None:
            msg = 'Please specify the input shape! E.g.:\n'
            msg += 'input_shape=(224, 224, 37)'
            print(msg)
            raise Exception(msg)
        else:
            self.input_shape = input_shape

        # Set regularization parameters
        # regularization for dense layers
        self.dense_reg = None
        if sum(dense_reg) != 0:
            self.dense_reg = tf.keras.regularizers.l1_l2(l1=dense_reg[0], l2=dense_reg[1])
        # Reg for conv layers
        self.conv_reg = None
        if sum(conv_reg) != 0:
            self.conv_reg = tf.keras.regularizers.l1_l2(l1=conv_reg[0], l2=conv_reg[1])
        # reg for bias
        self.bias_reg = None
        if bias_l2_reg != 0:
            self.bias_reg = tf.keras.regularizers.l2(bias_l2_reg)

        msg = 'Model {} selected!'.format(self.model_name)
        msg += '\nRegularization:'
        msg += '\nconv_l1_reg: {}, conv_l2_reg: {}'.format(conv_reg[0], conv_reg[1])
        msg += '\ndense_l1_reg: {}, dense_l2_reg: {}'.format(dense_reg[0], dense_reg[1])
        msg += '\nBias l2 reg: {}'.format(bias_l2_reg)
        self.log.info(msg)
        print(msg)

        self.model = None
        if self.model_name == 'baseline_CNN':
            self.model = self._get_baseline_CNN()

        elif self.model_name == 'ResNet50V2':
            self.model = self._get_ResNet50V2()

        elif self.model_name == 'Xception':
            self.model = self._get_Xception()

        elif self.model_name == 'Quick_test':
            self.model = self._get_Quick_test()

        elif self.model_name == 'Linear_Regression':
            self.model = self._get_Linear_Regression()

        else:
            msg = 'Specified model {} not implemented!'.format(self.model_name)
            self.log.error(msg)
            raise NotImplementedError(self.model_name)

        return self.model

    def _get_baseline_CNN(self):
        """
        Baseline Model
        Architecture:
        Input -> Conv3x3_64 -> BN -> ReLu -> MaxPool2x2
              -> Conv3x3_128 -> BN -> ReLu -> MaxPool2x2
              -> GlobalAveragePooling (Flatten filters by taking the average)
              -> Dense_256 -> BN -> ReLu
              -> Dense_128 -> BN -> ReLu
              -> Dense_1 (Prediction)
        """

        input_layer = tf.keras.Input(shape=self.input_shape, name='InputLayer')
        x = tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   kernel_regularizer=self.conv_reg,
                                   bias_regularizer=self.bias_reg,
                                   input_shape=self.input_shape
                                   )(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((2,2), strides=2)(x)

        x = tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same',
                                   #kernel_regularizer=self.conv_reg,
                                   #bias_regularizer=self.bias_reg
                                   )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((2,2), strides=2)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(units=256,
                                  kernel_regularizer=self.dense_reg,
                                  bias_regularizer=self.bias_reg
                                  )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(units=128,
                                  kernel_regularizer=self.dense_reg,
                                  bias_regularizer=self.bias_reg
                                  )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        prediction = tf.keras.layers.Dense(units=1,
                                           kernel_regularizer=self.dense_reg,
                                           bias_regularizer=self.bias_reg,
                                           )(x)
        # return model
        if self.return_custom_model:
            return CustomModel(inputs=input_layer, outputs=prediction)
        else:
            return tf.keras.models.Model(inputs=input_layer, outputs=prediction)


    def _get_Quick_test(self):
        """
        Model for quick tests in local machine (small architecture)
        Architecture:
        Input -> Conv3x3_64 -> BN -> ReLu -> MaxPool2x2
              -> GlobalAveragePooling (Flatten filters by taking the average)
              -> Dense_32 -> BN -> ReLu
              -> Dense_1 (Prediction)
        """

        input_layer = tf.keras.Input(shape=self.input_shape, name='InputLayer')
        x = tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   kernel_regularizer=self.conv_reg,
                                   bias_regularizer=self.bias_reg,
                                   input_shape=self.input_shape
                                   )(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((2,2), strides=2)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(units=32,
                                  kernel_regularizer=self.dense_reg,
                                  bias_regularizer=self.bias_reg
                                  )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        prediction = tf.keras.layers.Dense(units=1,
                                           kernel_regularizer=self.dense_reg,
                                           bias_regularizer=self.bias_reg,
                                           )(x)
        # return model
        if self.return_custom_model:
            return CustomModel(inputs=input_layer, outputs=prediction)
        else:
            return tf.keras.models.Model(inputs=input_layer, outputs=prediction)

    def _get_Linear_Regression(self):
        """
        Linear regresion model:
        Input -> GlobalAveragePooling (Per-Channel average)
              -> Dense_1 (Prediction)
        """

        input_layer = tf.keras.Input(shape=self.input_shape, name='InputLayer')

        x = tf.keras.layers.GlobalAveragePooling2D()(input_layer)

        prediction = tf.keras.layers.Dense(units=1,
                                           kernel_regularizer=self.dense_reg,
                                           bias_regularizer=self.bias_reg,
                                           )(x)
        # return model
        if self.return_custom_model:
            return CustomModel(inputs=input_layer, outputs=prediction)
        else:
            return tf.keras.models.Model(inputs=input_layer, outputs=prediction)

    def _get_ResNet50V2(self):
        """
        ResNet50V2 + Dense layers on the top.
        Architecture:
        Input -> ResNet50V2 without top layers (Dense prediction layers)
              -> GlobalAveragePooling (Flatten, output vector of size 2048)
              -> Dense_256 -> BN -> ReLu
              -> Dense_128 -> BN -> ReLu
              -> Dense_1 (Prediction)
        """

        # First, we need to load the RestNet50V2 arch. However, sine our input
        # shape is different from the original one, it is not possible to load
        # the arch. and the pretrained weights at the same time. Moreover, we
        # need to creat first a separate input layer with the shape of our data:
        input_layer = tf.keras.Input(shape=self.input_shape, name='InputLayer')

        # Now we load the base ResNet50V2 arch. using our input layer:
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=None,
            input_tensor=input_layer,
            pooling=None,
            classifier_activation=None
        )

        # load pretrained weights and biases
        if self.pre_training:
            base_model = self._set_ResNet50V2_pretrained_w_and_b(base_model)

        # For testing
        base_model = self._apply_regularization_to_prebuilt_model(base_model, 2)

        # Add some dense layers to predict the TR:
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(
            units=256,
            kernel_regularizer=self.dense_reg,
            bias_regularizer=self.bias_reg,
            #activity_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(
            units=128,
            kernel_regularizer=self.dense_reg,
            bias_regularizer=self.bias_reg,
            #activity_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        prediction = tf.keras.layers.Dense(
            units=1,
            kernel_regularizer=self.dense_reg,
            bias_regularizer=self.bias_reg,
            #activity_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)

        # return model
        if self.return_custom_model:
            return CustomModel(inputs=base_model.inputs, outputs=prediction)
        else:
            return tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

    def _get_Xception(self):
        """
        Xception + Dense layers on the top.
        Architecture:
        Input -> Xception without top layers (Dense prediction layers)
              -> GlobalAveragePooling (Flatten, output vector of size 2048)
              -> Dense_256 -> BN -> ReLu
              -> Dense_128 -> BN -> ReLu
              -> Dense_1 (Prediction)
        """

        # First, we need to load the arch. However, sine our input
        # shape is different from the original one, it is not possible to load
        # the arch. and the pretrained weights at the same time. Moreover, we
        # need to creat first a separate input layer with the shape of our data:
        input_layer = tf.keras.Input(shape=self.input_shape, name='InputLayer')

        # Now we load the base arch. using our input layer:
        base_model = tf.keras.applications.Xception(
            include_top=False,
            weights=None,
            input_tensor=input_layer,
            pooling=None,
            classifier_activation=None
        )

        if self.pre_training:
            base_model = self._set_Xception_pretrained_w_and_b(base_model)

        # For testing
        base_model = self._apply_regularization_to_prebuilt_model(base_model, 1)

        # Finally add some dense layers to predict the TR:
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(
            units=256,
            kernel_regularizer=self.dense_reg,
            bias_regularizer=self.bias_reg,
            #activity_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(
            units=128,
            kernel_regularizer=self.dense_reg,
            bias_regularizer=self.bias_reg,
            #activity_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        prediction = tf.keras.layers.Dense(
            units=1,
            kernel_regularizer=self.dense_reg,
            bias_regularizer=self.bias_reg,
            #activity_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)

        # return model
        if self.return_custom_model:
            return CustomModel(inputs=base_model.inputs, outputs=prediction)
        else:
            return tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

    def _set_Xception_pretrained_w_and_b(self, base_model):
        """
        This function returns a pretrained Base model (Xception)
        """

        # create the base pre-trained model
        pretrained_model = tf.keras.applications.Xception(
            weights="imagenet",
            include_top=False
        )

        # The next step is to save the pretrained weights in a temporary dict:
        pretrain_weights_and_biases = {}
        for layer in pretrained_model.layers:
            ws_and_bs = layer.get_weights()
            if len(ws_and_bs) != 0:
                pretrain_weights_and_biases[layer.name] = ws_and_bs

        # Before loading the pretrained weights and biases in our model, we need
        # to modify the weights of the first pretrained conv, since its original
        # shape is (3,3,3,32) (and wee need something like (3,3,31,32)).
        # To do this, we just avergae the pretrain weights over the third
        # dimesion, and then stack this mean convolution as many time as
        # necessary to fit the shape of our data:
        layer = pretrained_model.layers[1]
        l_name = layer.name # the name is "block1_conv1"
        pretrain_conv1_w = layer.get_weights()[0].copy()
        # create a conv of shape (3,3,1,32) averaging the dim 3:
        pretrain_conv1_w = pretrain_conv1_w.mean(axis=2).reshape(3,3,1,32)
        # how many times we will stack the mean conv:
        n_channels = self.input_shape[-1]
        # Create the weights of the firts conv
        conv1_conv_w_avg = pretrain_conv1_w
        for i in range(1, n_channels):
            conv1_conv_w_avg = np.concatenate((conv1_conv_w_avg, pretrain_conv1_w), axis=2)

        # Replace the origin pretrained weights for the first conv with the new
        pretrain_weights_and_biases[l_name] = [conv1_conv_w_avg]

        # Now, load the pretrained weights in our model:
        for layer in base_model.layers:
            l_name = layer.name
            if l_name in pretrain_weights_and_biases.keys():
                layer.set_weights(pretrain_weights_and_biases[l_name])

        return base_model


    def _set_ResNet50V2_pretrained_w_and_b(self, base_model):
        """
        This function returns a pretrained Base model (ResNet50V2)
        """

        # create the base pre-trained model
        pretrained_model = tf.keras.applications.ResNet50V2(
            weights="imagenet",
            include_top=False
        )

        # The next step is to save the pretrained weights in a temporary dict:
        pretrain_weights_and_biases = {}
        for layer in pretrained_model.layers:
            ws_and_bs = layer.get_weights()
            if len(ws_and_bs) != 0:
                pretrain_weights_and_biases[layer.name] = ws_and_bs

        # Before loading the pretrained weights and biases in our model, we need
        # to modify the weights of the first pretrained conv, since its original
        # shape is (7,7,3,64) (and wee need something like (7,7,31,64)).
        # To do this, we just avergae the pretrain weights over the third
        # dimesion, and then stack this mean convolution as many time as
        # necessary to fit the shape of our data:
        layer = pretrained_model.layers[2]
        l_name = layer.name # the name is "conv1_conv"
        pretrain_conv1_w = layer.get_weights()[0].copy()
        # create a conv of shape (7,7,1,64) averaging the dim 3:
        pretrain_conv1_w = pretrain_conv1_w.mean(axis=2).reshape(7,7,1,64)
        # how many times we will stack the mean conv:
        n_channels = self.input_shape[-1]
        # Create the weights of the firts conv
        conv1_conv_w_avg = pretrain_conv1_w
        for i in range(1, n_channels):
            conv1_conv_w_avg = np.concatenate((conv1_conv_w_avg, pretrain_conv1_w), axis=2)

        # Replace the origin pretrained weights for the first conv with the new
        pretrain_weights_and_biases[l_name] = [conv1_conv_w_avg, layer.get_weights()[1]]

        # Now, load the pretrained weights in the base model:
        for layer in base_model.layers:
            l_name = layer.name
            if l_name in pretrain_weights_and_biases.keys():
                layer.set_weights(pretrain_weights_and_biases[l_name])

        return base_model

    def _apply_regularization_to_prebuilt_model(self, base_model, layer_number):

        #https://towardsdatascience.com/how-to-add-regularization-to-keras-pre-trained-models-the-right-way-743776451091
        # Add regularization to a given layer (layer_number) of a prebuilt model
        prebuilt_layer = base_model.layers[layer_number]
        # add kernel reg
        if hasattr(prebuilt_layer, 'kernel_regularizer'):
            setattr(prebuilt_layer, 'kernel_regularizer', self.conv_reg)
        # add bias reg
        if hasattr(prebuilt_layer, 'bias_regularizer'):
            setattr(prebuilt_layer, 'bias_regularizer', self.bias_reg)
        # When we change the layers attributes, the change only happens in the model config file
        model_json = base_model.to_json()
        # Save the weights before reloading the model.
        model_w_and_b = copy.deepcopy(base_model.get_weights())
        # load the model from the config
        base_model = tf.keras.models.model_from_json(model_json)
        # Reload the model weights
        base_model.set_weights(model_w_and_b)

        return base_model

class Individual_Model_Training():
    """
    The idea of this clase is to join all the needed process to train a model into one class, so training several models using the same parameters can be easy.
    """
    def __init__(self):

        self.log = logging.getLogger(self.__class__.__name__)
        self._print_stdout_and_log("Individual_Model_Training class initialed")

        self.model = Predef_models()
        self.metrics = ['mse', 'mean_absolute_error']
        self.callbacks = []

    def init_model(self, arch_name='baseline_CNN', input_shape=(224, 224, 33), conv_reg=[0,0], dense_reg=[0,0], bias_l2_reg=0, pre_training=False, return_custom_model=True):

        self.model = self.model.select_model(model_name=arch_name,
                                             input_shape=input_shape,
                                             conv_reg=conv_reg,
                                             dense_reg=dense_reg,
                                             bias_l2_reg=bias_l2_reg,
                                             pre_training=pre_training,
                                             return_custom_model=return_custom_model
                                             )
        # Print model summary
        self.model.summary(print_fn=self._print_stdout_and_log)

        # Sanity check: print model losses (one for each layer regularized (1 for bias reg and 1 for kernel reg))
        self._print_stdout_and_log('Losses:\n{}'.format(self.model.losses))

    def build_model(self, loss_name='huber', learning_rate=0.001):

        # Select the loss function
        if loss_name == 'mse':
            loss = tf.keras.losses.MeanSquaredError()

        elif loss_name == 'huber':
            loss = tf.keras.losses.Huber(delta=1.0)

        elif loss_name == 'mean_absolute_error':
            loss = tf.keras.losses.MeanAbsoluteError()

        self._print_stdout_and_log('{} loss function selected. Building the model...'.format(loss_name))

        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=self.metrics
                           )
        self._print_stdout_and_log('Model compiled!')

    def fit_model(self, train_data, val_data, n_epochs, verbose_level):

        self._print_stdout_and_log('Starting model training...')

        # Save time before training
        tic = time.time()

        # Fit model
        self.history = self.model.fit(train_data,
                                      validation_data=val_data,
                                      epochs=n_epochs,
                                      callbacks=self.callbacks,
                                      verbose=verbose_level
                                      )
        toc = time.time()
        self._print_stdout_and_log('Training time (in mins): {}'.format((toc-tic)/60))

    def _print_stdout_and_log(self, msg):
        self.log.info(msg)
        print(msg)
