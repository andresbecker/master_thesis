import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import pandas as pd

class Predef_models():
    """
    This class contains different models
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Predef_models_and_utils class initialed')

    def select_model(self, model_name=None, input_shape=None, pre_training=False):

        self.model_name = model_name
        self.pre_training = pre_training

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

        elif self.model_name == 'ResNet50V2_R2':
            self.model = self._get_ResNet50V2_R2()

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
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

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
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        prediction = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)

        return model

    def _get_ResNet50V2_R2(self):

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
        prediction = tf.keras.layers.Dense(2)(x)

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
        if self.pre_training:
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
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

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

def plot_error_dist(df, y_true='y', y_models=['y_hat']):
    temp_df = df.copy()
    diff_names = ['set', 'perturbation']
    for y_model in y_models:
        # Column name for difference between y and y_model
        col_name = y_true+' - '+y_model
        diff_names.append(col_name)
        temp_df[col_name] = temp_df[y_true] - temp_df[y_model]

    temp_df = temp_df[diff_names].set_index(['set', 'perturbation'])
    temp_df = temp_df.stack().reset_index()
    #temp_df = temp_df.drop(['level_0'], axis=1)
    temp_df.columns = ['set', 'perturbation'] + ['diff_name', 'value']

    plt.figure(figsize=(20,7))
    sns.kdeplot(x='value',
                data=temp_df,
                hue='diff_name',
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

def plot_y_dist(df, y_true='y', y_hat='y_hat'):
    temp_df = df.copy()
    columns = [y_true, y_hat, 'mapobject_id_cell', 'set']
    temp_df = temp_df[columns]
    #temp_df = df.loc[:, df.columns != 'perturbation']
    temp_df = temp_df.set_index(['mapobject_id_cell', 'set']).stack().reset_index()
    temp_df.columns = ['mapobject_id_cell', 'set', 'var', 'value']

    plt.figure(figsize=(15,7))
    sns.boxplot(y='value',
                x='var',
                hue='set',
                data=temp_df)
    plt.title('Transcription Rate (TR) values distribution')

def plot_residuals(df=None, y_true='y', y_hat='y_hat'):

    df['diff'] = df[y_true] - df[y_hat]
    std = df['diff'].std()
    df['std_residuals'] = df['diff'] / std

    sns.scatterplot(
        data=df,
        x = y_true,
        y = 'std_residuals',
        hue = 'perturbation',
    )
    plt.hlines(y=2, xmin=np.min(df[y_true]), xmax=np.max(df[y_true]), color='red', ls='dashed')
    plt.hlines(y=-2, xmin=np.min(df[y_true]), xmax=np.max(df[y_true]), color='red', ls='dashed')
    plt.title(y_hat)

def plot_y_vs_y_hat(df, y_true='y', y_hat='y_hat'):
    sns.scatterplot(data=df,
                    x=y_true,
                    y=y_hat,
                    hue='perturbation',
                    #s=15,
                    alpha=0.5)
    plt.axis('equal')

    min_val = df[[y_true, y_hat]].min().values.min()
    max_val = df[[y_true, y_hat]].max().values.min()

    x_line = [min_val, max_val]
    y_line = x_line
    plt.plot(x_line, y_line, linestyle='dashed', color='red')
    plt.title(y_hat)

def get_BIC(y, y_hat, k):
    if k > 0:
        SS_residual = np.sum((y-y_hat)**2)
        n = y.shape[0]

        return n * np.log(SS_residual / n) + np.log(n) * k
    else:
        return 0

def get_metrics(df=None, k=0, y_true='y', y_hat='y_hat'):
    huber_loss = tf.keras.losses.Huber()

    # Create df to store metrics to compare models
    metric_columns = ['model', 'set', 'R2', 'BIC', 'MSE', 'MAE', 'Huber']
    temp_df = pd.DataFrame(columns=metric_columns)

    for ss in np.unique(df['set']):
        mask = (df['set'] == ss)
        y = df[y_true][mask].values
        y_hat_vals = df[y_hat][mask].values
        r2 = r2_score(y, y_hat_vals)
        bic = get_BIC(y=y, y_hat=y_hat_vals, k=k)
        mse = mean_squared_error(y, y_hat_vals)
        mae = mean_absolute_error(y, y_hat_vals)
        huber = huber_loss(y, y_hat_vals).numpy()
        temp_df = temp_df.append({'model':y_hat, 'set':ss, 'R2':r2, 'BIC':bic, 'MSE':mse, 'MAE':mae, 'Huber':huber}, ignore_index=True)

    return temp_df.round(4)

def plot_loss(history, metrics, p):
    keys = ['loss'] + metrics
    for i, key in enumerate(keys,1):
        warm_stage = int(p['number_of_epochs']*0.20)
        min_val = np.asarray(history[key]+history['val_'+key]).min()
        max_val = np.asarray(history[key][warm_stage:]+history['val_'+key][warm_stage:]).max()

        plt.subplot(3,1,i)
        plt.plot(history[key], label=key)
        plt.plot(history['val_'+key], label='val_'+key)
        val_min = np.asarray(history['val_'+key]).min()
        val_min_idx = np.argmin(history['val_'+key])
        label='bets val value\nEpoch={}\n{}={}'.format(val_min_idx,key,round(val_min,2))
        plt.scatter(x=val_min_idx, y=val_min, c='red', linewidths=4, label=label)
        plt.grid(True)
        plt.ylim([min_val, max_val])
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.legend()
        plt.title(key)
