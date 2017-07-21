from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.regularizers import l2, activity_l2
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.misc
import os
import sys
import pandas as pd
from collections import defaultdict
import binascii
from PIL import Image

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.02
set_session(tf.Session(config=config))

input_shape = (img_cols, img_rows, channel) = (30, 70, 1)
single_classes, char_len = 10, 4
nb_classes = single_classes * char_len
batch_size = 16
root_dir = "/root/pe_classify"


def convert2grey(pic_dir):
    pics = os.listdir(pic_dir)
    for pic in pics:
        im = Image.open(os.path.join(pic_dir, pic)).convert('L')
        im.save(os.path.join(root_dir, 'grey_pic', pic))


def get_data(pic_dir, img_cols=img_cols, img_rows=img_rows, channel=channel):
    pics = os.listdir(pic_dir)
    x = np.zeros([len(pics), img_cols, img_rows, channel])
    y = []
    for cc, pic in enumerate(pics):
        mat = scipy.misc.imread(os.path.join(pic_dir, pic)).astype(np.float32)
        x[cc, :, :, :] = mat
        split_int = list(os.path.splitext(pic)[0])
        y.append(split_int)
    df_y = pd.DataFrame(y)
    return x, df_y


def get_data_grey(pic_dir):
    pics = os.listdir(pic_dir)
    x = np.zeros([len(pics), img_cols, img_rows])
    y = []
    for cc, pic in enumerate(pics):
        mat = scipy.misc.imread(os.path.join(pic_dir, pic)).astype(np.float32)
        x[cc, :, :] = mat
        split_int = list(os.path.splitext(pic)[0])
        y.append(split_int)
    df_y = pd.DataFrame(y)
    return x, df_y


convert2grey(os.path.join(root_dir, "250_cap"))
x, y = get_data_grey(os.path.join(root_dir, "grey_pic"))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # , stratify=y)
print("x: %s, y: %s, x_train: %s, x_test: %s, y_train: %s, y_test: %s" % (
    x.shape, y.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape))

x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, channel)
x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, channel)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train_label = np_utils.to_categorical(y_train.iloc[:, 0], single_classes)
y_test_label = np_utils.to_categorical(y_test.iloc[:, 0], single_classes)
# print(y_train.iloc[:, 0])
# print(y_train_label)
# print(y_test.iloc[:, 0])
# print(y_test_label)
for i in range(1, char_len):
    # print y_train, "$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    tmp_y_train = np_utils.to_categorical(y_train.iloc[:, i], single_classes)
    tmp_y_test = np_utils.to_categorical(y_test.iloc[:, i], single_classes)
    y_train_label = np.c_[y_train_label, tmp_y_train]
    y_test_label = np.c_[y_test_label, tmp_y_test]

print("y_train_label: %s, y_test_label: %s" % (y_train_label.shape, y_test_label.shape))
y_train = y_train_label
y_test = y_test_label


# print(y_train_label[0:5])


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

    model.add(Convolution2D(nb_filters * 4, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters * 4, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(BatchNormalization(mode=2))
    # model.add(Convolution2D(nb_filters * 2 * 2, 3, 3,
    #                         border_mode='valid',
    #                         input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters * 2 * 2, 3, 3,
    #                         border_mode='valid',
    #                         input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(BatchNormalization(mode=2))
    # # model.add(Dropout(0.5))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))  # , W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


def cnn_model(nb_filters=32, kernel_size=(3, 3), pool_size=(2, 2), nb_epoch=12):
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    a = map(lambda x: map(int, x), y_train)
    b = map(lambda x: map(int, x), y_test)
    model.fit(x_train, a, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, shuffle=True, validation_split=0.1)
    score = model.evaluate(x_test, b, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    predict_labels = model.predict(x_test)
    a = map(lambda x: map(int, x), y_test)
    print a
    print predict_labels[0]


def single_model():
    model = make_model_2(dense_layer_sizes=[128, 128], nb_filters=32, nb_conv=3, nb_pool=2)
    model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, nb_epoch=50, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model


# model = single_model()
cnn_model()
