# -*- coding: utf-8 -*-

import os
import sys
import binascii
import csv
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model

batch_size = 64
nb_class = 10
epochs = 20
img_width, img_height = 130, 130


def img2array(filename):
    width = 28
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i: i + 2], 16) for i in range(0, len(hexst), 2)])
    fh = np.reshape(fh, (width, width))
    fh = np.uint8(fh)

    im = Image.fromarray(fh)
    im1 = im.resize((img_width, img_height), Image.ANTIALIAS)
    return np.array(im1)


def load_train_data(data_dir, train_label):
    os.chdir(data_dir)

    x = np.zeros([train_label.shape[0], img_width, img_height])
    for cc, filename in enumerate(train_label["md5"]):
        mat = img2array(filename).astype(np.float32)
        x[cc, :, :] = mat
    x = x.reshape(x.shape[0], img_width, img_height, 1)
    x /= 255
    y = to_categorical(train_label["type"], nb_class)
    return x, y


def load_pred_data(data_dir):
    os.chdir(data_dir)
    test_files = os.listdir(data_dir)
    x = np.zeros([len(test_files), img_width, img_height])
    md5s = []
    for i, filename in enumerate(test_files):
        md5s.append(filename)
        mat = img2array(filename).astype(np.float32)
        x[i, :, :] = mat
    x = x.reshape(x.shape[0], img_width, img_height, 1)
    x /= 255
    return x, md5s


train_csv = '/root/train.csv'
train_data_dir = '/root/train'
train_label = pd.read_csv(train_csv, header=None, names=('md5', 'type'))
x, y = load_train_data(train_data_dir, train_label)
x_pred, md5s = load_pred_data('/root/test')

input_tensor = Input(shape=(img_width, img_height, 1))
model = InceptionV3(input_tensor=input_tensor, weights=None, classes=nb_class)
# plot_model(model, to_file="incetionv3.png", show_shapes=True)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, validation_split=0.15, epochs=epochs, batch_size=batch_size)

model.load_weights("/root/weights130_18.h5")
y_pred = model.predict(x_pred)
y_pred1 = np.argmax(y_pred, 1)
y_pred2 = to_categorical(y_pred1, nb_class)

csv_pred = '/root/predict_submit_130.csv'
csvfile = file(csv_pred, 'wb')
writer = csv.writer(csvfile)
writer.writerows(map(lambda x: x[0].split() + list(x[1]), zip(md5s, y_pred2)))
csvfile.close()
