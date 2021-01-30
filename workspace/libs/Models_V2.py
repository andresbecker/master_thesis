import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import pandas as pd
import copy

class Predef_models():
    """
    This class contains different models
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Predef_models_and_utils class initialed')

    def select_model(self, model_name=None, input_shape=None, pre_training=False, conv_reg=[0,0], dense_reg=[0,0], bias_l2_reg=0):

        self.model_name = model_name
        self.pre_training = pre_training

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

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   kernel_regularizer=self.conv_reg,
                                   bias_regularizer=self.bias_reg,
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same',
                                   #kernel_regularizer=self.conv_reg,
                                   #bias_regularizer=self.bias_reg
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            #tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(
                units=256,
                kernel_regularizer=self.dense_reg,
                bias_regularizer=self.bias_reg,
                #activity_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(
                units=128,
                kernel_regularizer=self.dense_reg,
                bias_regularizer=self.bias_reg,
                #activity_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(
                units=1,
                kernel_regularizer=self.dense_reg,
                bias_regularizer=self.bias_reg,
                #activity_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
        ])

        return model


    def _get_Quick_test(self):
        """
        Model for quick tests in local machine (small architecture)
        Architecture:
        Input -> Conv3x3_64 -> BN -> ReLu -> MaxPool2x2
              -> GlobalAveragePooling (Flatten filters by taking the average)
              -> Dense_32 -> BN -> ReLu
              -> Dense_1 (Prediction)
        """

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   kernel_regularizer=self.conv_reg,
                                   bias_regularizer=self.bias_reg,
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                units=32,
                kernel_regularizer=self.dense_reg,
                bias_regularizer=self.bias_reg,
                #activity_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(
                units=1,
                kernel_regularizer=self.dense_reg,
                bias_regularizer=self.bias_reg,
                #activity_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
        ])

        return model


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

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model


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

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model

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
