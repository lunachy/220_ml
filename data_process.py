# -*- coding: utf-8 -*-

import os
import codecs
import json
import sys
import numpy as np
import hashlib
import scipy.misc
import shutil
import random
import pandas as pd
import csv

FILE_CHUNK_SIZE = 16 * 1024


def get_chunks(file_path):
    """Read file contents in chunks (generator)."""

    with open(file_path, "rb") as fd:
        while True:
            chunk = fd.read(FILE_CHUNK_SIZE)
            if not chunk: break
            yield chunk


def calc_md5(file_path):
    md5 = hashlib.md5()
    for chunk in get_chunks(file_path):
        md5.update(chunk)
    return md5.hexdigest()

train_dir = '/base/wangqh/machinelearning/mnist/train'
test_dir = '/base/wangqh/machinelearning/mnist/val'

train1_dir = '/root/mnist/train'
test1_dir = '/root/mnist/test'

train_bmp_dir = '/root/mnist/bmp_train'
test_bmp_dir = '/root/mnist/bmp_test'

train_csv = '/root/mnist/train.csv'
test_csv = '/root/mnist/test.csv'


def bmp_process(src_dir, dst_dir, bmp_dir, csv_path):
    data = []
    os.chdir(src_dir)
    for filename in os.listdir(src_dir):
        img_tuple = os.path.basename(filename).split('.')
        label, ext = img_tuple[0], img_tuple[2]
        if ext == "bmp":
            bin_file = img2bin(filename)
            file_md5 = calc_md5(bin_file)
            shutil.copy2(bin_file, os.path.join(dst_dir, file_md5))
            shutil.copy2(filename, os.path.join(bmp_dir, file_md5 + '.bmp'))
            data.append([file_md5, label])

    csvfile = file(csv_path, 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()


def img2bin(img):
    bin_file = img + '.bin'
    data = scipy.misc.imread(img)
    data.resize((data.shape[0] * data.shape[1]))
    with open(img + '.bin', 'wb') as f:
        f.write(data)
    return bin_file


bmp_process(train_dir, train1_dir, train_bmp_dir, train_csv)
bmp_process(test_dir, test1_dir, test_bmp_dir, test_csv)

# root_dir = "/root/round1-data-digit/"
# file_dir = os.path.join(root_dir, 'all')
# train_path = os.path.join(root_dir, 'train')
# test_path = os.path.join(root_dir, 'test')
# test_csv = os.path.join(root_dir, 'test.csv')
# train_csv = os.path.join(root_dir, 'train.csv')
# label = pd.read_csv(os.path.join(root_dir, "all.csv"))
# len_test = 10000
# os.chdir(file_dir)
# data = []  # 53560
# for file_path in os.listdir(file_dir):
#     item = label.loc[label["key"] == file_path]
#     if item['value'].any():
#         file_md5 = calc_md5(file_path)
#         data.append([file_md5, item.iloc[0, 1]])
#         shutil.copy2(file_path, os.path.join(train_path, file_md5))
#
# random.shuffle(data)
# for i, data1 in enumerate(data):
#     if i == len_test:
#         break
#     shutil.move(os.path.join(train_path, data1[0]), test_path)
#
# csvfile = file(test_csv, 'wb')
# writer = csv.writer(csvfile)
# writer.writerows(data[:len_test])
# csvfile.close()
#
# csvfile = file(train_csv, 'wb')
# writer = csv.writer(csvfile)
# writer.writerows(data[len_test:])
# csvfile.close()
