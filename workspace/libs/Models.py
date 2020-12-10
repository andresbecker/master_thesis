import tensorflow as tf
import logging
import numpy as np

class Predef_models():
    """
    This class contains different models
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Predef_models_and_utils class initialed')

    def select_model(self, model_name, input_shape=None):

        self.model_name = model_name
        if input_shape is None:
            msg = 'Please specify the input shape! E.g.:\n'
            msg += 'input_shape=(224, 224, 37)'
            print(msg)
            raise Exception(msg)
        else:
            self.input_shape = input_shape

        msg = '{} selected!'.format(self.model_name)
        self.log.info(msg)
        print(msg)

        self.model = None
        if self.model_name == 'baseline_CNN':
            self.model = self._get_baseline_CNN()

        elif self.model_name == 'small_CNN':
            self.model = self._get_small_CNN()

        elif self.model_name == 'baseline_CNN_w_Drop':
            self.model = self._get_baseline_CNN_w_Drop()

        elif self.model_name == '2ConvLey_2DensLey_w_BN_and_LeakyRelu':
            self.model = self._get_2ConvLey_2DensLey_w_BN_and_LeakyRelu()

        elif self.model_name == 'ResNet50V2':
            self.model = self._get_ResNet50V2()

        elif self.model_name == 'ResNet50V2_PreTrained':
            self.model = self._get_ResNet50V2_PreTrained()

        elif self.model_name == 'ResNet50V2_test':
            self.model = self._get_ResNet50V2_test()

        else:
            msg = 'Specified model {} not implemented!'.format(self.model_name)
            self.log.error(msg)
            raise NotImplementedError(self.model_name)

        return self.model

    def _get_baseline_CNN(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128),

            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_small_CNN(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.input_shape[-1], (3,3),
                                   padding='same',
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_baseline_CNN_w_Drop(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128),

            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_2ConvLey_2DensLey_w_BN_and_LeakyRelu(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(256, (3,3),
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_ResNet50V2(self):

        input_layer = tf.keras.Input(shape=self.input_shape,
            #batch_size=p['BATCH_SIZE'],
            name='InputLayer')

        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=None,
            input_tensor=input_layer,
            #input_shape=None,
            pooling=None,
            #classes=1000,
            classifier_activation=None,
            #classifier_activation='softmax',
            )

        x = base_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(128)(x)
        prediction = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model

    def _get_ResNet50V2_PreTrained(self):
        """
        Since we are using pretrained weights, then it is necessary two stages of training, one where the pretrained layers are freezed and a second one where the whole architecture is trained (with a small learning rate). Therefore, this function returns the model WITH THE PRETRAINED LAYERS FREZZED (i.e. pretrained_layer.trainable=False).
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

        # Now we need to load another ResNet50V2 arch, but now with the original
        # input shape and pretrained weights:
        pretrained_model = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classifier_activation=None,
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

        # Now, load the pretrained weights in our model:
        for layer in base_model.layers:
            l_name = layer.name
            if l_name in pretrain_weights_and_biases.keys():
                layer.set_weights(pretrain_weights_and_biases[l_name])

        # Now freez the pretrained layers for the first training (only train
        # last (dense) layers)
        for layer in base_model.layers:
            l_name = layer.name
            if l_name in pretrain_weights_and_biases.keys():
                layer.trainable = False
            else:
                layer.trainable = True

        # Finally add some dense layers to predict the TR:
        x = base_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(128)(x)
        prediction = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model

    def _get_ResNet50V2_test(self):

        input_layer = tf.keras.Input(shape=self.input_shape,
            #batch_size=p['BATCH_SIZE'],
            name='InputLayer')

        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            #weights="imagenet",
            weights=None,
            input_tensor=input_layer,
            #input_shape=None,
            pooling=None,
            #classes=1000,
            classifier_activation=None,
            #classifier_activation='softmax',
            )

        x = base_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(128)(x)
        prediction = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model
