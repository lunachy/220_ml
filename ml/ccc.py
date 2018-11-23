# coding=utf-8
import os
import time
from datetime import datetime
import sys
import logging
import pefile
import threading
import binascii
import multiprocessing
from math import sqrt
from multiprocessing import cpu_count, Pool
from queue import Queue
import cPickle as pickle
import csv
from PIL import Image
import json
from collections import Counter
from collections import defaultdict as ddict

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.utils.vis_utils import plot_model
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D,Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2,  l1
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import concatenate
from malware_modeling import Vocab,equal_tokens

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print device_lib.list_local_devices()

IMP_NAMES = []
#batch_size = 16
#epochs = 10
width = 167
img_rows, img_cols = [width] * 2
root_dir = "/root/pe_classify/"
pefile_dir = os.path.join(root_dir, '2017game_train')
train_csv = os.path.join(root_dir, '2017game_train.csv')
imp_name_path = os.path.join(root_dir, 'imp_names_map.dat')
train_data_path = os.path.join(root_dir, 'train_data.p')
opcode_dir = '/root/pe_classify/2017game_train_opcode'
data_op={}

def get_imp_name(target):
    imp_names = [1]
    try:
        pe = pefile.PE(target)
    except:
        # print("%s, not valid PE File" % target)
        return [1]

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imp_names.append(imp.name)
    return imp_names


def vectorize_imp_name(label):
    global IMP_NAMES
    row = [0] * width * width
    try:
        pe = pefile.PE(label[0])
    except:
        return [label[0], label[1], None]

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                row[IMP_NAMES.index(imp.name)] = 1
    # row.insert(0, label[1])
    return [label[0], label[1], row]  # return [md5, label, feature]

def preparing_X(dictionary,xid):
    vocab = Vocab()
    opcodes=set()
    for i,md5 in enumerate(xid):
        item=dictionary[md5]
        opcodes.update(item)
    print(len(opcodes))
    vocab.construct(opcodes)
    X= np.zeros([len(xid), 10000])
    for i,md5 in enumerate(xid):
        item=dictionary[md5]
        tokens=equal_tokens(item,maxlen=10000)
        encoded=[vocab.encode(opcode) for opcode in tokens]
        X[i,:]=encoded
    return X

def get_y(y_raw,threshold):
    choose=[]
    c=Counter(y_raw)
    for key in c.keys():
        num=c.get(key)
        if num >threshold:
            choose.append(key)
    y_new=["other" if x not in choose else x for x in y_raw]
    y_new=pd.Series(y_new,name="type")
    return choose,y_new

def get_categorical(y):
    y_num = pd.factorize(y)[0]
    n_class = len(set(y_num))
    Y = to_categorical(y_num, n_class)
    return Y

def getconvmodel(filter_length,nb_filter):
    model = Sequential()
    model.add(Convolution1D(filters=nb_filter,
                            input_shape=(10000,50),
                            kernel_size=filter_length,
                            padding='same',
                            activation='relu',
                            strides=1))
    #model.add(Lambda(sum_1d, output_shape=(nb_filter,)))
    #model.add(MaxPooling1D(pool_size=10000-filter_length+1))
    model.add(MaxPooling1D(pool_size=1000))
    model.add(Flatten())
    #model.add(BatchNormalization(mode=0))
    #model.add(Dropout(0.5))
    return model


def bag_of_convs_model(compile=True):
    main_input = Input(shape=(10000,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=50, input_dim=659, input_length=10000)(main_input)

    conv1 = getconvmodel(2, 64)(embedding)
    conv2 = getconvmodel(3, 64)(embedding)
    conv3 = getconvmodel(4, 64)(embedding)
    conv4 = getconvmodel(5, 64)(embedding)

    merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

    middle = Dense(256, activation='relu')(merged)
    middle = Dropout(0.5)(middle)

    middle = Dense(256, activation='relu')(middle)
    middle = Dropout(0.5)(middle)

    output = Dense(35, activation='softmax')(middle)

    model = Model(input=main_input, output=output)
    # adam=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if compile:
        model.compile(loss='categorical_crossentropy', optimizer="RMSprop", metrics=['accuracy'])
    return model


def combined_convs_model():
    input_op = Input(shape=(10000,), dtype='int32', name='input_op')
    input_pe = Input(shape=(167, 167, 1))

    embedding = Embedding(output_dim=50, input_dim=659, input_length=10000)(input_op)
    conv1 = getconvmodel(2, 64)(embedding)
    conv2 = getconvmodel(3, 64)(embedding)
    conv3 = getconvmodel(4, 64)(embedding)
    conv4 = getconvmodel(5, 64)(embedding)

    ins = Xception(include_top=False, weights=None, input_tensor=input_pe, input_shape=None, pooling="avg")(input_pe)

    merged = concatenate([conv1, conv2, conv3, conv4, ins], axis=1)

    middle = Dense(256, activation='relu')(merged)
    middle = Dropout(0.5)(middle)

    middle = Dense(256, activation='relu')(middle)
    middle = Dropout(0.5)(middle)

    output = Dense(35, activation='softmax')(middle)

    model = Model(input=[input_op, input_pe], output=output)
    # adam=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer="RMSprop", metrics=['accuracy'])
    return model


if __name__ == "__main__":
    os.chdir(pefile_dir)
    train_label = pd.read_csv(train_csv)
    nb_class = len(set(train_label["type"]))
    with open(imp_name_path, 'rb') as f:
        IMP_NAMES = list(pickle.load(f))
    print 'len(IMP_NAMES) {} ' .format(len(IMP_NAMES))
    start = time.time()
    pool = Pool(processes=cpu_count(), maxtasksperchild=400)
    # imp_names_list = pool.map(get_imp_name, os.listdir(pefile_dir))
    imp_names_vectorize = pool.map(vectorize_imp_name, zip(list(train_label["md5"]), list(train_label["type"])))
    pool.close()
    pool.join()
    end = time.time()
    print("cost all time: %s seconds." % (end - start))
    data_pe = pd.DataFrame(filter(lambda x: x[2] is not None, imp_names_vectorize), columns=['md5', 'type', 'imp_name'])
    print 'len(imp_names_vectorize){}, data_pe.shape {}'.format(len(imp_names_vectorize),data_pe.shape)
    idx_pe = data_pe["md5"]
    print 'len(idx_pe) {}'.format(len(idx_pe))

    os.chdir(opcode_dir)
    for i, jsonfile in enumerate(os.listdir(opcode_dir)):
        print(jsonfile)
        with open(jsonfile) as f:
            opcode = json.load(f)
        data_op.update(opcode)
    print 'len(data_op) {}'.format(len(data_op))

    xid = pickle.load(open('/root/pe_classify/xid_train.p'))
    inter_xid = list(set(xid).intersection(idx_pe))
    print'(len(xid)) {},(len(inter_xid)) {}' .format((len(xid)),(len(inter_xid)))
    X = preparing_X(data_op, inter_xid)
    X_op_md5 = pd.concat([pd.Series(inter_xid, name="md5"), pd.DataFrame(X)], axis=1)

    data_combine = pd.merge(data_pe, X_op_md5, on='md5')
    print 'data_combine.shape {}'.format(data_combine.shape)

    X_op = data_combine.iloc[:, 3:]
    print 'X_op.shape {}'.format(X_op.shape)
    a = data_combine['imp_name']
    X_pe = a.apply(pd.Series)
    X_pe = np.array(X_pe)
    X_pe_new = np.reshape(X_pe, (X_pe.shape[0], width, width, 1))
    print 'X_pe_new.shape {}'.format(X_pe_new.shape)

    y_raw = pickle.load(open(" /root/pe_classify/tmp_y.dat"))
    choose, _ = get_y(y_raw, 100)
    y = data_combine['type']
    y_new = pd.Series(["other" if x not in choose else x for x in y])
    Y = get_categorical(y_new)
    print 'Y.shape {}'.format(Y.shape)

    model = combined_convs_model()
    model.fit([np.array(X_op), X_pe_new], Y, validation_split=0.1, epochs=20, batch_size=16)