# coding:utf-8

import os
import time
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd

from malware_modeling import Vocab

opcode_dir = '/root/pe_classify/2017game_train_opcode'
asm_dir = '/root/pe_classify/2017game_train_asm'
root_dir = '/root/pe_classify/'
MAXLEN = 10000


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


def get_ops(asm):
    opcodes = []
    with open(asm, 'r') as f:
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

    return [os.path.splitext(asm)[0], opcodes]


def gen_x(ops_df):
    ops_set = reduce(lambda x, y: x | y, [set(ops) for ops in ops_df['opcodes']])

    vocab = Vocab()
    vocab.construct(ops_set)

    encode_ops = map(lambda ops_list: [vocab.encode(op) for op in ops_list], ops_df['opcodes'])
    return pd.DataFrame(encode_ops)


if __name__ == "__main__":
    os.chdir(asm_dir)
    # label = pd.read_csv(train_csv)

    start = time.time()
    pool = Pool(processes=cpu_count(), maxtasksperchild=400)
    ops = pool.map(get_ops, os.listdir(asm_dir)[:10])
    pool.close()
    pool.join()
    end = time.time()
    print("cost all time: %s seconds." % (end - start))

    ops_df = pd.DataFrame(ops, columns=['md5', 'opcodes'])
    print ops_df
    x = gen_x(ops_df)
    y = ops_df['md5']
    print x, y

