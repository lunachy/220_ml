# coding=utf-8
import logging.handlers
import os
import time
import sys
from multiprocessing import cpu_count, Pool
import pickle as cPickle
import numpy as np
import pandas as pd
import pefile
import signal
from malware_modeling import Vocab
import scipy.misc
from functools import reduce
from itertools import chain
from collections import Counter


IMP_NAMES = []
batch_size = 16
epochs = 50
train_size = 40000
train_size_2 = 47059
MAXLEN = 10000
OUTPUT_DIM = 50
#CPU_COUNT = cpu_count()  # 10
CPU_COUNT = 10


#root_dir = "F:/virus"
root_dir = "/data/root/pe_classify/"
root_dir_2 = "/root/pe_classify/"
pefile_train_dir = os.path.join(root_dir, '2017game_train')
pefile_test_dir = os.path.join(root_dir, '2017game_test')
asm_train_dir = os.path.join(root_dir, '2017game_train_asm')
asm_test_dir = os.path.join(root_dir, '2017game_test_asm')
pefile_unpack_train_dir = os.path.join(root_dir_2, 'unpack_train')
pefile_unpack_test_dir = os.path.join(root_dir_2, 'unpack_test')
asm_unpack_train_dir = os.path.join(root_dir_2, 'unpack_train_asm')
asm_unpack_test_dir = os.path.join(root_dir_2, 'unpack_test_asm')

imp_name_uncrypt_path = os.path.join(root_dir, 'imp_names_uncrypt_map.dat')
imp_name_path = os.path.join(root_dir, 'imp_names_map.dat')

ops_x_train_path = os.path.join(root_dir, 'ops_new.npz')
imp_x_train_path = os.path.join(root_dir, 'imp_new.npz')
ops_x_test_path = os.path.join(root_dir, 'ops_test_new.npz')
imp_x_test_path = os.path.join(root_dir, 'imp_test_new.npz')
ops_uncrypt_train_path = os.path.join(root_dir, 'ops_uncrypt_train.npz')
ops_uncrypt_test_path = os.path.join(root_dir, 'ops_uncrypt_test.npz')
imp_uncrypt_train_path = os.path.join(root_dir, 'imp_uncrypt_train.npz')
imp_uncrypt_test_path = os.path.join(root_dir, 'imp_uncrypt_test.npz')
ops_combined_train_path = os.path.join(root_dir, 'ops_combined_train.npz')
ops_combined_test_path = os.path.join(root_dir, 'ops_combined_test.npz')
imp_combined_train_path = os.path.join(root_dir, 'imp_combined_train.npz')
imp_combined_test_path = os.path.join(root_dir, 'imp_combined_test.npz')
ops_3g_train_path = os.path.join(root_dir, 'ops_3g_train.npz')
ops_3g_test_path = os.path.join(root_dir, 'ops_3g_test.npz')
ops_4g_train_path = os.path.join(root_dir, 'ops_4g_train.npz')
ops_4g_test_path = os.path.join(root_dir, 'ops_4g_test.npz')

train_csv = os.path.join(root_dir, '2017game_train.csv')
test_csv = os.path.join(root_dir, '2017game_test.csv')
train_uncrypt_csv = os.path.join(root_dir, 'train_uncrypt.csv')
test_uncrypt_csv = os.path.join(root_dir, 'test_uncrypt.csv')
train_md5_sig_encoding_path = os.path.join(root_dir, 'train_md5_sig_encoding.npz')
test_md5_sig_encoding_path = os.path.join(root_dir, 'test_md5_sig_encoding.npz')


log = logging.getLogger()
formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

fh = logging.handlers.WatchedFileHandler(os.path.join(root_dir, 'pe_analyzer.log'))
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def filter_array(ops):
    return filter(lambda op:op!=1,ops)

def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

def get_ngrams_index(ops_ngram):
    row=[0]*len(opcodes_list)
    d=dict(Counter(ops_ngram))
    for k in d.keys():
        if k in opcodes_list:
            row[opcodes_list.index(k)]=d[k]
    return row

if __name__ == "__main__":
    # ####################full data####################################
    # train_data = pd.read_csv(train_csv)
    # label_dic = dict(zip(train_data["md5"], train_data["type"]))
    # asm_files_train = os.listdir(asm_train_dir)
    # md5s_train = [md5 for md5 in train_data['md5'] if (md5 + ".asm") in asm_files_train]
    # # train_label = train_label[train_label['md5'].isin(md5s_train)]
    # train_label = [label_dic[md5] for md5 in md5s_train]
    # test_data = pd.read_csv(test_csv)
    # label_dic_test = dict(zip(test_data["md5"], test_data["type"]))
    # asm_files_test = os.listdir(asm_test_dir)
    # md5s_test = [md5 for md5 in test_data['md5'] if (md5 + ".asm") in asm_files_test]
    # test_label = [label_dic_test[md5] for md5 in md5s_test]
    # print(len(md5s_train), len(md5s_test))
    # print(len(train_label), len(test_label))
    #
    # ############load imp data####################
    # imp_x_train_combined = np.load(imp_x_train_path)["arr_0"]
    # imp_x_train_combined = np.array(imp_x_test_path)
    # imp_x_test_combined = np.load(imp_combined_test_path)["arr_0"]
    # imp_x_test_combined = np.array(imp_x_test_combined)
    # print(imp_x_train_combined.shape, imp_x_test_combined.shape)
    # # width=167
    # width = 161
    # imp_x_train_combined = np.reshape(imp_x_train_combined, (imp_x_train_combined.shape[0], width * width))
    # imp_x_test_combined =np.reshape(imp_x_test_combined ,(imp_x_test_combined .shape[0],width*width))
    # print(imp_x_train_combined.shape, imp_x_test_combined.shape)
    #
    ##########load ops data##################
    ops_x_train_combined = np.load(ops_x_train_path)["arr_0"]
    ops_x_train_combined = np.array(ops_x_train_combined)
    ops_x_test_combined = np.load(ops_x_test_path)["arr_0"]
    ops_x_test_combined = np.array(ops_x_test_combined)
    print(ops_x_train_combined.shape, ops_x_test_combined.shape)
    #
    # #####y label###############
    # nb_class = len(set(train_label))
    # label = sorted(set(train_label))
    # dic = dict(zip(label, range(0, nb_class)))
    # y_combined = [dic[y] for y in train_label]
    # # y_combined = to_categorical(y_combined, nb_class)
    # y_test_combined = [dic[y] for y in test_label]
    # # y_test_combined = to_categorical(y_test_combined, nb_class)
    # # print(y_combined.shape,y_test_combined.shape)
    # print(len(y_combined), len(y_test_combined)
    #
    # ###########add opcodes##################
    # ops_x_train_combined = map(lambda ops: filter_array(ops), ops_x_train_combined)
    # ops_x_test_combined = map(lambda ops: filter_array(ops), ops_x_test_combined)
    # ngram_range = 3
    # max_features = 1000
    # print(len(ops_x_train_combined), 'train sequences')
    # print(len(ops_x_test_combined), 'test sequences')
    # print('Average train sequence length: {}'.format(np.mean(list(map(len, ops_x_train_combined)), dtype=int)))
    # print('Average test sequence length: {}'.format(np.mean(list(map(len, ops_x_test_combined)), dtype=int)))
    # if ngram_range > 1:
    #     print('Adding {}-gram features'.format(ngram_range))
    #     # Create set of unique n-gram from the training set.
    #     ngram_set = set()
    #     for input_list in ops_x_train_combined:  ###
    #         for i in range(2, ngram_range + 1):
    #             set_of_ngram = create_ngram_set(input_list, ngram_value=i)
    #             ngram_set.update(set_of_ngram)
    # print(len(ngram_set))
    # # Dictionary mapping n-gram token to a unique integer.
    # # Integer values are greater than max_features in order
    # # to avoid collision with existing features.
    # start_index = max_features + 1
    # token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    # indice_token = {token_indice[k]: k for k in token_indice}
    # # max_features is the highest integer that could be found in the dataset.
    # max_features = np.max(list(indice_token.keys())) + 1
    # print(max_features)
    # # Augmenting x_train and x_test with n-grams features
    # ops_x_train_3g = add_ngram(ops_x_train_combined, token_indice, ngram_range)
    # ops_x_test_3g = add_ngram(ops_x_test_combined, token_indice, ngram_range)
    # print('Average train sequence length: {}'.format(np.mean(list(map(len, ops_x_train_3g)), dtype=int)))
    # print('Average test sequence length: {}'.format(np.mean(list(map(len, ops_x_test_3g)), dtype=int)))
    # total_ops_3g_list = list(chain(*ops_x_train_3g))
    # print(len(total_ops_3g_list))
    # ops_3g_d = dict(Counter(total_ops_3g_list))
    # print(len(ops_3g_d))
    # opcodes_list = [k for k, v in ops_3g_d.items() if ops_3g_d[k] > 100]
    # print(len(opcodes_list))
    #
    # # start = time.time()
    # # pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    # # ops_x_train_3g_counter = pool.map(get_ngrams_index, ops_x_train_3g)
    # # pool.close()
    # # pool.join()
    # # end = time.time()
    # # log.info("get_ops_x_train_3g_counter cost %.2f seconds." % (end - start))
    #
    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    # ops_x_test_3g_counter = pool.map(get_ngrams_index, ops_x_test_3g)
    # pool.close()
    # pool.join()
    # end = time.time()
    # log.info("get_ops_x_test_3g_counter cost %.2f seconds." % (end - start))
    #
    # # ops_x_train_3g_counter = np.array(ops_x_train_3g_counter)
    # ops_x_test_3g_counter = np.array(ops_x_test_3g_counter)
    # print(ops_x_test_3g_counter.shape)
    # # np.savez_compressed(ops_3g_train_path, ops_x_train_3g_counter)
    # np.savez_compressed(ops_3g_test_path, ops_x_test_3g_counter)
    # print('gen imp_x finished')


    # ###########add 4-grams opcodes##################
    ops_x_train_combined = map(lambda ops: filter_array(ops), ops_x_train_combined)
    ops_x_test_combined = map(lambda ops: filter_array(ops), ops_x_test_combined)
    ngram_range = 4
    max_features = 1000
    print(len(ops_x_train_combined), 'train sequences')
    print(len(ops_x_test_combined), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, ops_x_train_combined)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, ops_x_test_combined)), dtype=int)))
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in ops_x_train_combined:  ###
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
    print(len(ngram_set))
    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1
    print(max_features)
    # Augmenting x_train and x_test with n-grams features
    ops_x_train_4g = add_ngram(ops_x_train_combined, token_indice, ngram_range)
    ops_x_test_4g = add_ngram(ops_x_test_combined, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, ops_x_train_4g)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, ops_x_test_4g)), dtype=int)))
    total_ops_4g_list = list(chain(*ops_x_train_4g))
    print(len(total_ops_4g_list))
    ops_4g_d = dict(Counter(total_ops_4g_list))
    print(len(ops_4g_d))
    opcodes_list = [k for k, v in ops_4g_d.items() if ops_4g_d[k] > 100]
    print(len(opcodes_list))

    start = time.time()
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    ops_x_train_4g_counter = pool.map(get_ngrams_index, ops_x_train_4g)
    pool.close()
    pool.join()
    end = time.time()
    log.info("get_ops_x_train_3g_counter cost %.2f seconds." % (end - start))

    start = time.time()
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    ops_x_test_4g_counter = pool.map(get_ngrams_index, ops_x_test_4g)
    pool.close()
    pool.join()
    end = time.time()
    log.info("get_ops_x_test_3g_counter cost %.2f seconds." % (end - start))

    ops_x_train_4g_counter = np.array(ops_x_train_4g_counter)
    ops_x_test_4g_counter = np.array(ops_x_test_4g_counter)
    print(ops_x_train_4g_counter,ops_x_test_4g_counter.shape)
    np.savez_compressed(ops_4g_train_path, ops_x_train_4g_counter)
    np.savez_compressed(ops_4g_test_path, ops_x_test_4g_counter)
    print('gen imp_x finished')