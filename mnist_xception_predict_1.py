# -*- coding: utf-8 -*-

import os
import sys
import binascii
from collections import Counter
from datetime import datetime
import itertools
from keras.applications.xception import Xception

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.utils.np_utils import to_categorical
import csv
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import classification_report, confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# print device_lib.list_local_devices()

batch_size = 16
nb_class = 10
epochs = 50
img_rows, img_cols = [299] * 2


def bin2array(filename):
    width = 28
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = np.array([int(hexst[i: i + 2], 16) for i in range(0, len(hexst), 2)])  # 按字节分割
    fh = np.reshape(fh, (width, width))  # 根据设定的宽度生成矩阵
    fh = np.uint8(fh)

    im = Image.fromarray(fh)
    resize_img = im.resize((img_rows, img_cols), Image.ANTIALIAS)
    resize_fh = np.array(resize_img)
    return resize_fh


def load_predict_data(data_dir):
    os.chdir(data_dir)
    files = os.listdir(data_dir)
    x = np.zeros([len(files), img_rows, img_cols])
    md5s = []
    for cc, filename in enumerate(files):
        md5s.append(filename)
        mat = bin2array(filename).astype(np.float32)
        x[cc, :, :] = mat
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x /= 255
    return x, md5s


x_predict, md5s = load_predict_data('/root/mnist/test')
print "Load data successfully!"

input_tensor = Input(shape=(img_rows, img_cols, 1))
model = Xception(input_tensor=input_tensor, weights=None, classes=nb_class)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
weights_dir = '/root/mnist/better_weights'
os.chdir(weights_dir)
weights = sorted(os.listdir(weights_dir))
y_all = []


def most_common(x):
    x2 = Counter(x).most_common(2)
    if len(x2) < 2:
        return x2[0][0]
    # only 0/1 different return the bigger, otherwise return the smaller
    if x2[0][0] == 6 or x2[1][0] == 6:
        return 6
    elif x2[0][0] == 9 or x2[1][0] == 9:
        return 9
    else:
        return x2[0][0]


for i, weight in enumerate(weights):
    print datetime.now(), weight
    model.load_weights(weight)
    y_predict = model.predict(x_predict)
    y_pred = np.argmax(y_predict, 1)
    y_all.append(y_pred)

    y_pred = map(most_common, np.array(y_all).T)
    # y_pred = [Counter(tmp_y).most_common(1)[0][0] for tmp_y in np.array(y_all).T]
    y_predict1 = to_categorical(y_pred, nb_class)

    csvfile = file('/root/mnist/predict_new_submit_{}.csv'.format(i), 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(map(lambda x: x[0].split() + list(x[1]), zip(md5s, y_predict1)))
    csvfile.close()

