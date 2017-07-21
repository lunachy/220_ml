# -*- coding: utf-8 -*-

import os
import sys
import binascii
from keras.applications.xception import Xception
from keras.layers import Input
from keras.utils.np_utils import to_categorical
import csv
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_size = 16
nb_class = 10
epochs = 30
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


x_predict, md5s = load_predict_data('/root/test')
print "Load data successfully!"

input_tensor = Input(shape=(img_rows, img_cols, 1))
model = Xception(input_tensor=input_tensor, weights=None, classes=nb_class)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("weights.h5")

y_predict = model.predict(x_predict)
y_pred = np.argmax(y_predict, 1)
y_predict1 = to_categorical(y_pred, nb_class)

csv_pred = '/root/predict_submit.csv'
csvfile = file(csv_pred, 'wb')
writer = csv.writer(csvfile)
writer.writerows(map(lambda x: x[0].split() + list(x[1]), zip(md5s, y_predict1)))
csvfile.close()
