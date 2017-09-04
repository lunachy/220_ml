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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print device_lib.list_local_devices()

IMP_NAMES = []
batch_size = 16
epochs = 10
width = 167

root_dir = "/root/pe_classify/"
pefile_dir = os.path.join(root_dir, '2017game_train')
train_csv = os.path.join(root_dir, '2017game_train.csv')
imp_name_path = os.path.join(root_dir, 'imp_names_map.dat')
ops_x_path = os.path.join(root_dir, 'ops.npz')
imp_x_path = os.path.join(root_dir, 'imp.npz')
train_data_path = os.path.join(root_dir, 'train_data.p')
ops_md5_path = os.path.join(root_dir, 'ops_md5s.npz')

opcode_dir = '/root/pe_classify/2017game_train_opcode'
asm_dir = '/root/pe_classify/2017game_train_asm'
MAXLEN = 10000
OUTPUT_DIM = 50
INPUT_DIM = 0
MAX_NB_WORDS = 0
CPU_COUNT = cpu_count()

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


def words_encode(word_list):
    tokenizer = Tokenizer(MAX_NB_WORDS)
    tokenizer.fit_on_texts(word_list)
    sequences = tokenizer.texts_to_sequences(word_list)
    return sum(sequences, [])


def gen_x(ops_list):
    # global MAX_NB_WORDS
    global INPUT_DIM
    ops_set = reduce(lambda x, y: x | y, [set(ops) for ops in ops_list])
    # MAX_NB_WORDS = len(ops_set) + 1
    #
    # encode_ops = map(words_encode, ops_list)
    # return np.array(encode_ops)

    vocab = Vocab()
    INPUT_DIM = vocab.construct(ops_set)

    # TODO: modify encoding
    encode_ops = map(lambda ops: [vocab.encode(op) for op in ops], ops_list)
    return np.array(encode_ops)


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
    row = [0] * width * width
    try:
        pe = pefile.PE(md5)
    except:
        print("%s, not valid PE File" % md5)
        return None

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                row[IMP_NAMES.index(imp.name)] = 1
    return row


def getconvmodel(filter_length, nb_filter):
    model = Sequential()
    model.add(Convolution1D(filters=nb_filter,
                            input_shape=(MAXLEN, OUTPUT_DIM),
                            kernel_size=filter_length,
                            padding='same',
                            activation='relu',
                            strides=1))
    model.add(MaxPooling1D(pool_size=1000))
    model.add(Flatten())
    return model


def combined_convs_model():
    input_op = Input(shape=(MAXLEN,), dtype='int32', name='input_op')
    input_pe = Input(shape=(width, width, 1))

    embedding = Embedding(output_dim=OUTPUT_DIM, input_dim=INPUT_DIM, input_length=MAXLEN)(input_op)
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

    output = Dense(nb_class, activation='softmax')(middle)

    model = Model(input=[input_op, input_pe], output=output)

    model.compile(loss='categorical_crossentropy', optimizer="RMSprop", metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # get intersection of opcodes and import funcs
    log.info('start read_csv')
    train_label = pd.read_csv(train_csv)
    asm_files = os.listdir(asm_dir)
    # TODO: modify
    md5s = [md5 for md5 in train_label['md5'] if (md5 + ".asm") in asm_files]
    train_label = train_label[train_label['md5'].isin(md5s)]
    log.info('train_label.shape: %s', train_label.shape)
    nb_class = len(set(train_label["type"]))

    # process opcode
    os.chdir(asm_dir)
    start = time.time()
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    ops = pool.map(get_ops, md5s)
    log.info('get ops finished!')
    np.savez_compressed(ops_md5_path, md5s, ops)
    pool.close()
    pool.join()
    end = time.time()
    log.info("get_ops cost %.2f seconds." % (end - start))
    sys.exit()

    # ops_x = gen_x(ops)
    # np.savez_compressed(ops_x_path, ops_x)
    ops_x = np.load(ops_x_path)['arr_0']
    log.info('gen ops_x finished')

    # process import table
    os.chdir(pefile_dir)

    # start = time.time()
    # imp_names_list = pool.map(get_imp_name, md5s)
    # pool.close()
    # pool.join()
    # end = time.time()
    # log.info("get_imp_name cost %.2f seconds." % (end - start))
    with open(imp_name_path, 'rb') as f:
        IMP_NAMES = list(cPickle.load(f))
        IMP_NAMES.remove(1)
    log.info('get IMP_NAMES successfully, length of IMP_NAMES: %s', len(IMP_NAMES))
    # IMP_NAMES = reduce(lambda x, y: x | y, [set(imp_names) for imp_names in imp_names_list])
    # log.info('get IMP_NAMES successfully, length of IMP_NAMES: %s', len(IMP_NAMES))
    # sys.exit()

    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    # imp_names_vectorize = pool.map(vectorize_imp_name, md5s)
    # pool.close()
    # pool.join()
    # end = time.time()
    # log.info("vectorize_imp_name cost %.2f seconds." % (end - start))
    #
    # imp_x = np.array(imp_names_vectorize)
    # imp_x = np.reshape(imp_x, (imp_x.shape[0], width, width, 1))
    # np.savez_compressed(imp_x_path, imp_x)
    imp_x = np.load(imp_x_path)['arr_0']
    log.info('gen imp_x finished')

    imp_y = pd.factorize(train_label['type'])
    imp_y = to_categorical(imp_y[0], nb_class)
    log.info('imp_x.shape: %s, ops_x.shape: %s, imp_y.shape: %s' % (imp_x.shape, ops_x.shape, imp_y.shape))

    # model = combined_convs_model()
    # model.fit([ops_x, imp_x], imp_y, validation_split=0.1, epochs=epochs, batch_size=batch_size)
    sys.exit()

    input_tensor = Input(shape=(width, width, 1))
    model = Xception(input_tensor=input_tensor, weights=None, classes=nb_class)
    # plot_model(model, to_file="xception.png", show_shapes=True)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.load_weights(os.path.join(root_dir, 'weights_09.h5'))
    save_best_callback = ModelCheckpoint(os.path.join(root_dir, 'weights_{epoch:02d}.h5'), monitor='val_acc',
                                         mode='max', save_best_only=False, save_weights_only=True)
    model.fit(imp_x, imp_y, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[save_best_callback])
    # model.fit(imp_x, imp_y, validation_split=0.15, epochs=epochs, batch_size=batch_size)
