# coding=utf-8
import logging.handlers
import os
import time
import sys
from multiprocessing import cpu_count, Pool
import cPickle
import numpy as np
import pandas as pd
import pefile
import signal
import tensorflow as tf
from keras.applications.xception import Xception
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from tensorflow.python.client import device_lib
from keras.preprocessing.text import Tokenizer

from malware_modeling import Vocab
from functools import reduce

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print device_lib.list_local_devices()

IMP_NAMES = []
batch_size = 16
epochs = 10
# width = 167
# width = 161
complete_flag = False
unpack_flag = True

root_dir = "/data/root/pe_classify/"
root_dir_2 = "/root/pe_classify/"
# root_dir = "F:/virus/"
pefile_train_dir = os.path.join(root_dir, '2017game_train')
pefile_test_dir = os.path.join(root_dir, '2017game_test')
asm_train_dir = os.path.join(root_dir, '2017game_train_asm')
asm_test_dir = os.path.join(root_dir, '2017game_test_asm')
pefile_unpack_train_dir = os.path.join(root_dir_2, 'unpack_train')
pefile_unpack_test_dir = os.path.join(root_dir_2, 'unpack_test')
asm_unpack_train_dir = os.path.join(root_dir_2, 'unpack_train_asm')
asm_unpack_test_dir = os.path.join(root_dir_2, 'unpack_test_asm')
imp_name_path = os.path.join(root_dir, 'imp_names_map.dat')
imp_name_uncrypt_path = os.path.join(root_dir, 'imp_names_uncrypt_map.dat')
imp_name_combined_path = os.path.join(root_dir, 'imp_names_combined_map.dat')
ops_x_train_path = os.path.join(root_dir, 'ops_new.npz')
imp_x_train_path = os.path.join(root_dir, 'imp_new.npz')
ops_x_test_path = os.path.join(root_dir, 'ops_test_new.npz')
imp_x_test_path = os.path.join(root_dir, 'imp_test_new.npz')
# opcodes_major_train_path = os.path.join(root_dir, 'opcodes_major_train.npz')
# opcodes_major_test_path = os.path.join(root_dir, 'opcodes_major_test.npz')
# ops_major_train_path = os.path.join(root_dir, 'ops_major_train.npz')
# ops_major_test_path = os.path.join(root_dir, 'ops_major_test.npz')
ops_uncrypt_train_path = os.path.join(root_dir, 'ops_uncrypt_train.npz')
ops_uncrypt_test_path = os.path.join(root_dir, 'ops_uncrypt_test.npz')
imp_uncrypt_train_path = os.path.join(root_dir, 'imp_uncrypt_train.npz')
imp_uncrypt_test_path = os.path.join(root_dir, 'imp_uncrypt_test.npz')
# ops_unpack_train_path = os.path.join(root_dir, 'ops_unpack_train.npz')
# ops_unpack_test_path = os.path.join(root_dir, 'ops_unpack_test.npz')
# imp_unpack_train_path = os.path.join(root_dir, 'imp_unpack_train.npz')
# imp_unpack_test_path = os.path.join(root_dir, 'imp_unpack_test.npz')
ops_combined_train_path = os.path.join(root_dir, 'ops_combined_train.npz')
ops_combined_test_path = os.path.join(root_dir, 'ops_combined_test.npz')
imp_combined_train_path = os.path.join(root_dir, 'imp_combined_train.npz')
imp_combined_test_path = os.path.join(root_dir, 'imp_combined_test.npz')

train_csv = os.path.join(root_dir, '2017game_train.csv')
test_csv = os.path.join(root_dir, '2017game_test.csv')
train_uncrypt_csv = os.path.join(root_dir, 'train_uncrypt.csv')
test_uncrypt_csv = os.path.join(root_dir, 'test_uncrypt.csv')
train_unpack_csv = os.path.join(root_dir, 'train_unpack.csv')
test_unpack_csv = os.path.join(root_dir, 'test_unpack.csv')

BOUNDARY = '; ---------------------------------------------------------------------------'
MAXLEN = 10000
OUTPUT_DIM = 50
INPUT_DIM = 0
MAX_NB_WORDS = 0
CPU_COUNT = 2  # cpu_count()-1

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


def isvalid(s):
    bytes = '0123456789abcdef'
    if len(s) == 2:
        if s[0] in bytes:
            return False  # ins cannot have these
    if not s.isalpha():
        return False
    if s[0].isupper():
        return False
    if s in ['align', 'extrn', 'unicode', 'assume', 'offset']:
        return False
    return True


def get_ops(md5):
    opcodes = []
    with open(md5 + '.asm', 'r') as f:
        for line in f:
            if not line.startswith('\t'):
                continue
            try:
                opcode = line.strip().split()[0]
            except IndexError:
                continue
            if isvalid(opcode):
                opcodes.append(opcode)
                if len(opcodes) >= MAXLEN:
                    break

    if len(opcodes) < MAXLEN:
        opcodes = [0] * (MAXLEN - len(opcodes)) + opcodes

    return opcodes


def get_call_ops(md5):
    opcodes_str = ''
    with open(md5 + '.asm', 'r') as f:
        for line in f:
            if line.strip() == BOUNDARY:
                opcodes_str += ';'
            if not line.startswith('\t'):
                continue
            try:
                opcode = line.strip().split()[0]
            except IndexError:
                continue
            if isvalid(opcode):
                opcodes_str = opcodes_str + '_' + opcode

    opcodes_list = opcodes_str.split(';')
    # filter section where there is no call function
    opcodes_f = filter(lambda op: op.find('call') != -1, opcodes_list)
    # opcodes = filter(lambda op: op, '_'.join(opcodes_f).split('_'))
    if '_'.join(opcodes_f).replace('_', ''):
        opcodes = filter(lambda op: op, '_'.join(opcodes_f).split('_'))
    else:
        opcodes = filter(lambda op: op, '_'.join(opcodes_list).split('_'))
    if len(opcodes) >= MAXLEN:
        opcodes = opcodes[0:MAXLEN]
    else:
        opcodes = [0] * (MAXLEN - len(opcodes)) + opcodes
    return opcodes


def words_encode(word_list):
    tokenizer = Tokenizer(MAX_NB_WORDS)
    tokenizer.fit_on_texts(word_list)
    sequences = tokenizer.texts_to_sequences(word_list)
    return sum(sequences, [])


def gen_x_test(ops_list_1, ops_list_2):
    # global MAX_NB_WORDS
    global INPUT_DIM
    ops_set = reduce(lambda x, y: x | y, [set(ops) for ops in ops_list_1])

    vocab = Vocab()
    INPUT_DIM = vocab.construct(ops_set)

    encode_ops_train = map(lambda ops: [vocab.encode(op) for op in ops], ops_list_1)
    encode_ops_test = map(lambda ops: [vocab.encode(op) for op in ops], ops_list_2)
    return np.array(encode_ops_train), np.array(encode_ops_test)


def get_imp_name(target):
    imp_names = []
    try:
        pe = pefile.PE(target)
    except:
        # print("%s, not valid PE File" % target)
        return imp_names

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imp_names.append(imp.name)
    return imp_names


def vectorize_imp_name(md5):
    # row = [0] * width * width
    row = [0] * len(IMP_NAMES)
    try:
        pe = pefile.PE(md5)
    except:
        print("%s, not valid PE File" % md5)
        # return None
        return row

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                row[IMP_NAMES.index(imp.name)] = 1
    return row


def vectorize_imp_name_test(md5):
    # row = [0] * width * width
    row = [0] * len(IMP_NAMES)
    try:
        pe = pefile.PE(md5)
    except:
        print("%s, not valid PE File" % md5)
        # return None
        return row

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name in IMP_NAMES:
                    row[IMP_NAMES.index(imp.name)] = 1
    return row


def get_data(path, method, md5s):
    os.chdir(path)
    start = time.time()
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    data_list = pool.map(method, md5s)
    print(len(data_list))
    pool.close()
    pool.join()
    end = time.time()
    log.info("%s cost %.2f seconds." % ("method", end - start))
    return data_list


if __name__ == "__main__":
    # get intersection of opcodes and import funcs
    log.info('start read_csv')
    train_label = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    asm_files_train = os.listdir(asm_train_dir)
    asm_files_test = os.listdir(asm_test_dir)
    md5s_train = [md5 for md5 in train_label['md5'] if (md5 + ".asm") in asm_files_train]
    md5s_test = [md5 for md5 in test_data['md5'] if (md5 + ".asm") in asm_files_test]
    # train_label = train_label[train_label['md5'].isin(md5s_train)]
    # test_data = test_data[test_data['md5'].isin(md5s_test)]
    log.info('train_label.shape: %s', train_label.shape)
    log.info('train_label.shape: %s', test_data.shape)

    ############uncrpyt#######################
    ##########################################
    train_uncrypt = pd.read_csv(train_uncrypt_csv)
    test_uncrypt = pd.read_csv(test_uncrypt_csv)
    md5s_train_uncrypt = [md5 for md5 in train_uncrypt["md5"] if md5 in md5s_train]
    md5s_test_uncrypt = [md5 for md5 in test_uncrypt["md5"] if md5 in md5s_test]
    print(len(md5s_train_uncrypt), len(md5s_test_uncrypt))
    # train_label_uncrypt = train_uncrypt[train_uncrypt['md5'].isin( md5s_train_uncrypt)]
    # test_data_uncrypt = test_uncrypt[test_uncrypt['md5'].isin(md5s_test_uncrypt)]
    # print(train_label_uncrypt.shape,test_data_uncrypt.shape)
    # index_train_un = train_label["md5"].isin(un_train)
    # index_test_un = test_data["md5"].isin(un_test)
    # print(sum(index_train_un), sum(index_test_un))

    ############unpack#######################
    ##########################################
    # train_unpack = pd.read_csv(train_unpack_csv)
    # test_unpack = pd.read_csv(test_unpack_csv)
    train_unpack = os.listdir(pefile_unpack_train_dir)
    test_unpack = os.listdir(pefile_unpack_test_dir)
    train_unpack = map(lambda file: file.split("_")[1], train_unpack)
    test_unpack = map(lambda file: file.split("_")[1], test_unpack)
    asm_files_unpack_train = os.listdir(asm_unpack_train_dir)
    asm_files_unpack_test = os.listdir(asm_unpack_test_dir)
    asm_files_unpack_train = map(lambda file: file.split("_")[1].split(".")[0], asm_files_unpack_train)
    asm_files_unpack_test = map(lambda file: file.split("_")[1].split(".")[0], asm_files_unpack_test)
    md5s_train_unpack = [md5 for md5 in train_unpack if md5 in md5s_train]
    md5s_test_unpack = [md5 for md5 in test_unpack if md5 in md5s_test]
    md5s_train_unpack = [md5 for md5 in md5s_train_unpack if md5 in asm_files_unpack_train]
    md5s_test_unpack = [md5 for md5 in md5s_test_unpack if md5 in asm_files_unpack_test]
    print(len(md5s_train_unpack), len(md5s_test_unpack))
    # train_label_unpack = train_unpack[train_unpack['md5'].isin(md5s_train_unpack)]
    # test_data_unpack = test_uncrypt[test_unpack['md5'].isin(md5s_test_unpack)]
    # print(train_label_unpack.shape, test_data_unpack.shape)

    ##########################
    ##### process opcode######
    # os.chdir(asm_train_dir)
    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    # #ops_train = pool.map(get_ops, md5s_train)
    # pool.close()
    # pool.join()
    # end = time.time()
    # log.info("get_ops_train_uncrypt cost %.2f seconds." % (end - start))
    # #np.savez_compressed(opcodes_major_train_path, md5s_train, opcodes_train)
    # #print len(opcodes_train)
    # #print len(filter(lambda x: x, opcodes_train))

    opcodes_train_uncrypt = get_data(asm_train_dir, get_ops, md5s_train_uncrypt) if complete_flag == False \
        else get_data(asm_train_dir, get_ops, md5s_train)
    opcodes_test_uncrypt = get_data(asm_test_dir, get_ops, md5s_test_uncrypt) if complete_flag == False \
        else get_data(asm_test_dir, get_ops, md5s_test)
    opcodes_train_unpack = get_data(asm_unpack_train_dir, get_ops, md5s_train_unpack) if unpack_flag == True else []
    opcodes_test_unpack = get_data(asm_unpack_test_dir, get_ops, md5s_test_unpack) if unpack_flag == True else []
    opcodes_train_uncrypt.extend(opcodes_train_unpack)
    opcodes_train = opcodes_train_uncrypt
    opcodes_test_uncrypt.extend(opcodes_test_unpack)
    opcodes_test = opcodes_test_uncrypt
    print(len(opcodes_train), len(opcodes_test))
    # #
    # #
    ops_x_train, ops_x_test = gen_x_test(opcodes_train, opcodes_test)
    np.savez_compressed(ops_combined_train_path, ops_x_train)
    np.savez_compressed(ops_combined_test_path, ops_x_test)
    log.info('INPUT_DIM:%s', ops_x_train.shape)
    log.info('INPUT_DIM:%s', ops_x_test.shape)
    log.info('gen ops_x finished')

    ################################
    ##### process import table#####
    # os.chdir(pefile_train_dir)
    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    # imp_names_list = pool.map(get_imp_name, md5s_train_un)
    # pool.close()
    # pool.join()
    # end = time.time()
    # log.info("get_imp_name cost %.2f seconds." % (end - start))

    imp_names_list_uncrypt = get_data(pefile_train_dir, get_imp_name, md5s_train_uncrypt) if complete_flag == False \
        else get_data(pefile_train_dir, get_imp_name, md5s_train)
    imp_names_list_unpack = get_data(pefile_unpack_train_dir, get_imp_name,
                                     md5s_train_unpack) if unpack_flag == True else []
    imp_names_list_uncrypt.extend(imp_names_list_unpack)
    imp_names_list = imp_names_list_uncrypt

    IMP_NAMES = reduce(lambda x, y: x | y, [set(imp_names) for imp_names in imp_names_list])
    with open(os.path.join(root_dir, 'imp_names_combined_map.dat'), 'wb') as f:
        cPickle.dump(IMP_NAMES, f)
    log.info('get IMP_NAMES successfully, length of IMP_NAMES: %s', len(IMP_NAMES))

    # with open(imp_name_combined_path, 'rb') as f:
    #     IMP_NAMES = list(cPickle.load(f))
    #     #IMP_NAMES.remove(1)
    # log.info('get IMP_NAMES successfully, length of IMP_NAMES: %s', len(IMP_NAMES))
    # #


    # os.chdir(pefile_train_dir)
    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    # imp_names_vectorize_train = pool.map(vectorize_imp_name, md5s_train_un)
    # print(len(imp_names_vectorize_train))
    # pool.close()
    # pool.join()
    # end = time.time()
    # log.info("vectorize_imp_name_train cost %.2f seconds." % (end - start))

    imp_names_vectorize_train_uncrypt = get_data(pefile_train_dir, vectorize_imp_name,
                                                 md5s_train_uncrypt) if complete_flag == False \
        else get_data(pefile_train_dir, vectorize_imp_name, md5s_train)
    imp_names_vectorize_train_unpack = get_data(pefile_unpack_train_dir, vectorize_imp_name,
                                                md5s_train_unpack) if unpack_flag == True else []
    imp_names_vectorize_test_uncrypt = get_data(pefile_test_dir, vectorize_imp_name_test,
                                                md5s_test_uncrypt) if complete_flag == False \
        else get_data(pefile_test_dir, vectorize_imp_name, md5s_test)
    imp_names_vectorize_test_unpack = get_data(pefile_unpack_test_dir, vectorize_imp_name_test,
                                               md5s_test_unpack) if unpack_flag == True else []
    # #
    imp_names_vectorize_train_uncrypt.extend(imp_names_vectorize_train_unpack)
    imp_names_vectorize_train = imp_names_vectorize_train_uncrypt
    imp_x_train = np.array(imp_names_vectorize_train)
    # imp_x_train = np.reshape(imp_x_train, (imp_x_train.shape[0], width, width, 1))
    print(imp_x_train.shape)
    np.savez_compressed(imp_combined_train_path, imp_x_train)
    # #
    imp_names_vectorize_test_uncrypt.extend(imp_names_vectorize_test_unpack)
    imp_names_vectorize_test = imp_names_vectorize_test_uncrypt
    imp_x_test = np.array(imp_names_vectorize_test)
    # imp_x_test = np.reshape(imp_x_test, (imp_x_test.shape[0], width, width, 1))
    print(imp_x_test.shape)
    np.savez_compressed(imp_combined_test_path, imp_x_test)
    log.info('gen imp_x finished')
    sys.exit()

    # imp_y = pd.factorize(train_label['type'])
    # imp_y = to_categorical(imp_y[0], nb_class)
    # log.info('imp_x.shape: %s, ops_x.shape: %s, imp_y.shape: %s' % (imp_x.shape, ops_x.shape, imp_y.shape))

    # model = combined_convs_model()
    # model.fit([ops_x, imp_x], imp_y, validation_split=0.1, epochs=epochs, batch_size=batch_size)

    # input_tensor = Input(shape=(width, width, 1))
    # model = Xception(input_tensor=input_tensor, weights=None, classes=nb_class)
    # plot_model(model, to_file="xception.png", show_shapes=True)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.load_weights(os.path.join(root_dir, 'weights_09.h5'))
    # save_best_callback = ModelCheckpoint(os.path.join(root_dir, 'weights_{epoch:02d}.h5'), monitor='val_acc',
    # mode='max', save_best_only=False, save_weights_only=True)
    # model.fit(imp_x, imp_y, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[save_best_callback])
    # model.fit(imp_x, imp_y, validation_split=0.15, epochs=epochs, batch_size=batch_size)
