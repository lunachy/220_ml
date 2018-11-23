# -*- coding: utf-8 -*-

import os
import sys
import binascii
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
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

print device_lib.list_local_devices()

batch_size = 8
nb_class = 10
epochs = 20
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


def load_train_data(data_dir, label):
    os.chdir(data_dir)

    x = np.zeros([label.shape[0], img_rows, img_cols])
    for cc, filename in enumerate(label["md5"]):
        mat = bin2array(filename).astype(np.float32)
        x[cc, :, :] = mat
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x /= 255
    y = to_categorical(label["type"], nb_class)
    return x, y


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


csv_path = '/root/mnist/train.csv'
train_data_dir = '/root/mnist/train'
label = pd.read_csv(csv_path, header=None, names=('md5', 'type'))
x, y = load_train_data(train_data_dir, label)
x_predict, md5s = load_predict_data('/root/mnist/test')
print "Load data successfully!"

input_tensor = Input(shape=(img_rows, img_cols, 1))
model = Xception(input_tensor=input_tensor, weights=None, classes=nb_class)
# plot_model(model, to_file="xception.png", show_shapes=True)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights("/root/mnist/weights299_9968.h5")
save_best_callback = ModelCheckpoint('/root/mnist/weights299_{epoch:02d}.h5', monitor='val_acc',
                                     mode='max', save_best_only=False, save_weights_only=True)
model.fit(x, y, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[save_best_callback])
