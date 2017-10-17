#!/usr/bin/python
# coding=utf-8
import functools
import logging.handlers
import os
import MySQLdb
import json
from collections import Counter, OrderedDict
import logging.handlers
import time
import sys
from multiprocessing import cpu_count, Pool
import cPickle
import numpy as np
import pandas as pd
import pefile
import signal
import argparse
from itertools import chain, combinations
import types
import gc
import binascii
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# from tensorflow.python.client import device_lib
#
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.25
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
#
# device_lib.list_local_devices()

log_dir = '/data/log/'

CPU_COUNT = 15  # cpu_count()
train_count = 47067
test_count = 9398
total_count = train_count + test_count

log = logging.getLogger()
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

fh = logging.handlers.WatchedFileHandler(os.path.join(log_dir, os.path.splitext(__file__)[0] + '.log'))
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)


def log_decorate(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        try:
            return_value = func(*args, **kw)
        except Exception as e:
            return_value = []
            log.info('error msg: %s', e)
            # log.info('error msg: %s, args: %s, kw: %s' %(e, args, kw))
        end = time.time()
        log.info('Called func[%s], starts: %s, costs %.2f seconds.' %
                 (func.__name__, time.strftime('%H:%M:%S', time.localtime(start)), (end - start)))
        return return_value

    return wrapper


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pool_map(func, iter_obj):
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    return_list = pool.map(func, iter_obj)
    pool.close()
    pool.join()
    return return_list


def reduce_list_set(list_set):
    return reduce(lambda x, y: x | y, list_set)


def reduce_list_list(list_list):
    return reduce(lambda x, y: x | y, map(lambda l: set(l), list_list))


def reduce_list_counter(list_counter):
    return reduce(lambda x, y: x | y, map(lambda c: set(c.keys()), list_counter))


def _encode_set(value_set, all_types):
    row = [0] * len(all_types)
    for v in value_set:
        if v in all_types:
            row[all_types.index(v)] = 1
    return row


def multi_encoding_l_s(list_set, filter_count=1):
    """
    one-hot encoding: list_set -> [0, 1]
    :param list_set
    :param filter_count
    :return: encoded result
    """
    # all_values = reduce_list_set(list_set)
    # [{1, 3, 5}, {1, 3, 7}] --> Counter({1: 2, 3: 2, 5: 1, 7: 1})
    _all_types = Counter(sum([list(s) for s in list_set], []))
    # if threshold = 1, Counter({1: 5, 3: 3, 5: 1, 7: 2}) --> [1, 3, 7]
    all_types = filter(lambda c: _all_types[c] > filter_count, _all_types)

    log.info('[multi_encoding_l_s] length of all_types: %s', len(all_types))
    _encode_p = functools.partial(_encode_set, all_types=all_types)
    return pool_map(_encode_p, list_set)


def _encode_counter(value_counter, all_types):
    row = [0] * len(all_types)
    for v in value_counter:
        if v in all_types:
            row[all_types.index(v)] = value_counter[v]
    return row


def multi_encoding_l_c(list_input, filter_count=0):
    """
    one-hot encoding: list_input -> [0, 1, 2, 3……]
    :param list_input
    :param filter_count
    :return: encoded result
    """
    all_types = []
    print list_input[0]
    if isinstance(list_input[0], types.ListType):
        all_count = Counter(list(chain(*list_input)))
        # [Counter({1: 3, 3: 1, 5: 1}), Counter({1: 2, 3: 2, 7: 2})] --> Counter({1: 5, 3: 3, 5: 1, 7: 2})
        # all_count = reduce(lambda c1, c2: c1 + c2, list_count)
        # if threshold = 1, Counter({1: 5, 3: 3, 5: 1, 7: 2}) --> [1, 3, 7]
        log.info('get all_types from Counter')
        all_types = filter(lambda c: all_count[c] > filter_count, all_count)
        list_count = [Counter(i) for i in list_input]
    elif isinstance(list_input[0], dict):
        log.info('get all_types from list_counter')
        # all_types = list(reduce(lambda x, y: x | y, map(lambda c: set(c.keys()), list_input)))
        _types = map(lambda c: c.keys(), list_input)
        _key_count = Counter(list(chain(*_types)))
        all_types = filter(lambda x: _key_count[x] > filter_count, _key_count)
        list_count = list_input
    else:
        log.warning('no matching para!')

    assert all_types, 'all_types should not be []'
    log.info('[multi_encoding_l_c] length of all_types: %s', len(all_types))
    _encode_p = functools.partial(_encode_counter, all_types=all_types)
    return pool_map(_encode_p, list_count)
