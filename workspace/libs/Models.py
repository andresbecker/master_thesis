import tensorflow as tf
import logging

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

        if self.model_name == 'baseline_CNN_2':
            self.model = self._get_baseline_CNN_2()

        elif self.model_name == '2ConvLey_2DensLey_w_BN_and_LeakyRelu':
            self.model = self._get_2ConvLey_2DensLey_w_BN_and_LeakyRelu()

        elif self.model_name == 'baseline_CNN_w_BN_Drop':
            self.model = self._get_baseline_CNN_w_BN_Drop()

        elif self.model_name == 'small_CNN':
            self.model = self._get_small_CNN()

        elif self.model_name == 'ResNet50V2':
            self.model = self._get_ResNet50V2()

        elif self.model_name == 'ResNet50V2_PreTrained':
            self.model = self._get_ResNet50V2_PreTrained()

        else:
            msg = 'Specified model {} not implemented!'.format(self.model_name)
            self.log.error(msg)
            raise NotImplementedError(self.model_name)

        return self.model

    def _get_baseline_CNN(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same',
                                   activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_baseline_CNN_2(self):

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
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_small_CNN(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.input_shape[-1], (3,3),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        return model

    def _get_baseline_CNN_w_BN_Drop(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3),
                                   padding='same',
                                   kernel_initializer='glorot_normal',
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(128, (3,3),
                                   padding='same',
                                   kernel_initializer='glorot_normal'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
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
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        prediction = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model

    def _get_ResNet50V2_PreTrained(self):

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
        x = tf.keras.layers.Dense(128)(x)
        prediction = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model
