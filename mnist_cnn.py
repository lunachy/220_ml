'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
import tensorflow as tf
from tensorflow.python.client import device_lib
import cv2
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


print device_lib.list_local_devices()

batch_size = 16
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = [150]*2
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = X_train[:10000], y_train[:10000]
X_train = np.array(map(lambda x: cv2.resize(x, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC), X_train))
X_test = np.array(map(lambda x: cv2.resize(x, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC), X_test))

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
with open(__file__):  # tf.device('/gpu:1'):
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    model = Xception(input_tensor=input_tensor, weights=None, classes=nb_classes)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_split=0.2, epochs=nb_epoch, batch_size=batch_size)

    # model = Sequential()
    #
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    #                         border_mode='valid',
    #                         input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    #           verbose=1, shuffle=True, validation_split=0.2)
    score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
