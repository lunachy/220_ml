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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def gen_data(data_dir, label, batch_size=batch_size):
    os.chdir(data_dir)
    y_all = to_categorical(label["type"], nb_class)
    label_length = label.shape[0]
    i = 0
    while i < label_length:
        if label_length - i < batch_size:
            batch_size = label_length - i
        x = np.zeros([batch_size, img_rows, img_cols])
        y = y_all[i: i + batch_size]
        for cc, filename in enumerate(label["md5"][i: i + batch_size]):
            mat = bin2array(filename).astype(np.float32)
            x[cc, :, :] = mat
        x = x.reshape(x.shape[0], img_rows, img_cols, 1)
        x /= 255
        i += batch_size
        yield x, y


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
weights_dir = '/root/mnist/weights'
os.chdir(weights_dir)
weights = sorted(os.listdir(weights_dir))
y_all = []
a = pd.read_csv('/root/mnist/test.csv', header=None, names=['md5', 'type'])
a1 = a.sort_values(['md5'])


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


result = [a1['md5'], a1['type']]
for i, weight in enumerate(weights):
    print datetime.now(), weight
    model.load_weights(weight)
    y_predict = model.predict(x_predict)
    y_pred = np.argmax(y_predict, 1)
    y_all.append(y_pred)

    # y_pred = map(most_common, np.array(y_all).T)
    y_pred = [Counter(tmp_y).most_common(1)[0][0] for tmp_y in np.array(y_all).T]
    y_predict1 = to_categorical(y_pred, nb_class)

    csvfile = file('/root/mnist/predict_most_{}.csv'.format(i), 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(zip(md5s, y_pred))
    csvfile.close()

    csvfile = file('/root/mnist/predict_most_submit_{}.csv'.format(i), 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(map(lambda x: x[0].split() + list(x[1]), zip(md5s, y_predict1)))
    csvfile.close()

    b = pd.read_csv('/root/mnist/predict_most_{}.csv'.format(i), header=None, names=['md5', 'type'])
    b1 = b.sort_values(['md5'])
    # indice = np.array(map(np.array_equal, a1['type'], b1['type'])) == False
    result.append(b1['type'])
    # print np.array([a1['md5'][indice], a1['type'][indice], b1['type'][indice]]).T
    ratio = np.mean(map(np.array_equal, a1['type'], b1['type']))
    print "accuracy: ", i, ratio
    # print confusion_matrix(a1['type'], b1['type'])
sys.exit()


def predict_wrong(rt_row):
    rt_row = rt_row[1:]
    return not (np.array(rt_row) == rt_row[0]).all()


rt = np.array(result).T
wrong_result = rt[map(predict_wrong, rt)]
csvfile = file('/root/mnist/wrong_result_all.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerows(wrong_result)
csvfile.close()

sys.exit()

y_pred = [Counter(i).most_common(1)[0][0] for i in np.array(y_all).T]
y_predict1 = to_categorical(y_pred, nb_class)

csv_pred = '/root/mnist/predict.csv'
csvfile = file(csv_pred, 'wb')
writer = csv.writer(csvfile)
# writer.writerows(map(lambda x: x[0].split() + list(x[1]), zip(md5s, y_predict1)))
writer.writerows(zip(md5s, y_pred))
csvfile.close()

csv_pred = '/root/mnist/predict_submit.csv'
csvfile = file(csv_pred, 'wb')
writer = csv.writer(csvfile)
writer.writerows(map(lambda x: x[0].split() + list(x[1]), zip(md5s, y_predict1)))
csvfile.close()

a = pd.read_csv('/root/mnist/test.csv', header=None, names=['md5', 'type'])
b = pd.read_csv('/root/mnist/predict.csv', header=None, names=['md5', 'type'])
a1 = a.sort_values(['md5'])['type']
b1 = b.sort_values(['md5'])['type']
ratio = np.mean(map(np.array_equal, a1, b1))
print "accuracy: ", ratio
print confusion_matrix(a1, b1)
