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

batch_size = 4
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


csv_path = '/root/mnist/train.csv'
train_data_dir = '/root/mnist/train'
label = pd.read_csv(csv_path, header=None, names=('md5', 'type'))
x, y = load_train_data(train_data_dir, label)
x_predict, md5s = load_predict_data('/root/mnist/test')

test_csv_path = '/root/mnist/test.csv'
test_data_dir = '/root/mnist/test'
label1 = pd.read_csv(test_csv_path, header=None, names=('md5', 'type'))
x1, y1 = load_train_data(test_data_dir, label1)
print "Load data successfully!"

input_tensor = Input(shape=(img_rows, img_cols, 1))
model = Xception(input_tensor=input_tensor, weights=None, classes=nb_class)
# plot_model(model, to_file="xception.png", show_shapes=True)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("/root/mnist/weights299_9968.h5")
save_best_callback = ModelCheckpoint('/root/mnist/weights299_{epoch:02d}.h5', monitor='val_acc',
                                     mode='max', save_best_only=False, save_weights_only=True)
# model.fit(x, y, validation_data=(x1, y1), epochs=epochs, batch_size=batch_size, callbacks=[save_best_callback])
# model.fit_generator(gen_data(train_data_dir, label), steps_per_epoch=label.shape[0]//batch_size - batch_size, epochs=epochs,
#                     validation_data=gen_data(test_data_dir, label1), validation_steps=label1.shape[0]//batch_size - batch_size,
#                     callbacks=[save_best_callback])

# sys.exit()
y_predict = model.predict(x_predict)
y_pred = np.argmax(y_predict, 1)

csv_pred = '/root/mnist/predict.csv'
csvfile = file(csv_pred, 'wb')
writer = csv.writer(csvfile)
writer.writerows(zip(md5s, y_pred))
csvfile.close()

a = pd.read_csv('/root/mnist/test.csv', header=None, names=['md5', 'type'])
b = pd.read_csv('/root/mnist/predict.csv', header=None, names=['md5', 'type'])
a1 = a.sort_values(['md5'])['type']
b1 = b.sort_values(['md5'])['type']
ratio = np.mean(map(np.array_equal, a1, b1))
print "accuracy: ", ratio
print confusion_matrix(a1, b1)
