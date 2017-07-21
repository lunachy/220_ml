from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import numpy as np
import scipy.misc
import os
import pandas as pd
from collections import defaultdict
import binascii
from PIL import Image

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


def get_data(data_set, label, width):
    X = np.zeros([label.shape[0], width, width])
    for cc, x in enumerate(label.iloc[:, 0]):
        image = data_set + "VirusShare_" + x + ".jpg"
        mat = scipy.misc.imread(image).astype(np.float32)
        X[cc, :, :] = mat
    return X


label = pd.read_csv('/root/pe_classify/train_label.csv')
X = get_data("/root/pe_classify/pic/", label, 512)
label["Virus_type2"] = pd.factorize(label.iloc[:, 1])[0]
y = label["Virus_type2"]
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

img_rows, img_cols = 512, 512
input_shape = (512, 512, 1)
nb_classes = 20

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print('X_test  shape', X_test.shape)

from keras.utils import np_utils
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(Y_train.shape)
print(Y_test.shape)

from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.regularizers import l2, activity_l2


def make_model_2(dense_layer_sizes, nb_filters, nb_conv, nb_pool):
    '''Creates model comprised of 2 convolutional layers followed by dense layers
    dense_layer_sizes: List of layer sizes. This list has one number for each layer
    nb_filters: Number of convolutional filters in each convolutional layer
    nb_conv: Convolutional kernel size
    nb_pool: Size of pooling area for max pooling
    '''

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=input_shape,
                            init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters * 2, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters * 2, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.5))

    # model.add(BatchNormalization(mode=2))
    model.add(Convolution2D(nb_filters * 2 * 2, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters * 2 * 2, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(BatchNormalization(mode=2))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


def single_model():
    model = make_model_2(dense_layer_sizes=[128], nb_filters=8, nb_conv=5, nb_pool=2)
    model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, nb_epoch=10, shuffle=True, verbose=1)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model


model = single_model()
