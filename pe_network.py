# coding=utf-8
import os
import logging
import pefile
import cPickle
import time
import sys
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from keras.utils.np_utils import to_categorical
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
from PIL import Image

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import classification_report, confusion_matrix
from malware_modeling import Vocab, equal_tokens

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print device_lib.list_local_devices()


def gen_x(ops_dict, md5s):
    opcodes = reduce(lambda x, y: x | y, [set(ops_dict[md5]) for md5 in md5s])
    print(len(opcodes))

    vocab = Vocab()
    vocab.construct(opcodes)

    ops_len = 100000

    x = np.zeros([len(md5s), ops_len])
    for i, md5 in enumerate(md5s):
        item = ops_dict[md5]
        tokens = equal_tokens(item, ops_len)
        encoded = [vocab.encode(opcode) for opcode in tokens]
        x[i, :] = encoded
        # if i==0:
        # break
    # X.set_index=xid
    return x
