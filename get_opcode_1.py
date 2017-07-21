# coding:utf-8

import os
import time
from datetime import datetime
import threading
import binascii
import multiprocessing
from math import sqrt
from multiprocessing import cpu_count
from queue import Queue
import cPickle

from PIL import Image
import numpy
import json
import sys

opcode_dir = '/root/pe_classify/2017game_test_opcode'
asm_dir = '/base/wangqh/machinelearning/microsoft_malware/train_asm'
os.chdir(asm_dir)
RESULT = {}


def isvalid(s):
    bytes = '0123456789abcdef'
    if len(s) == 2:
        if s[0] in bytes:
            return False  # ins cannot have these
    if not s.isalpha():
        return False
    if s[0].isupper():
        return False
    if s in ['align', 'extrn', 'unicode', 'assume', 'offset', 'public']:
        return False
    if s.lower() != s:
        return False
    return True


# for cc, fx in enumerate(xid):
#     f = open(data_path + '/' + fx + '.asm')
#     loc = {}  # address -> instruction
#     for line in f:
#         if '.text' != line[:5] and '.code' != line[:5]:
#             # most of ins are in those two parts
#             continue
#         xx = line.split()
#         if len(xx) > 2:
#             add = xx[0].split(':')[1]  # address
#             for i in xx[1:]:
#                 if isvalid(i):  # get the first token that is not a byte
#                     loc[add] = i
#                     break  # one instruction per line (address)

for i, asm in enumerate(os.listdir(asm_dir), 1):
    print i
    opcodes = []
    if i % 100 == 0:
        OPS = set()
        # OPS = reduce(lambda x, y: x | y, [set(j) for j in RESULT.values()])
        # print OPS, len(OPS)
        print i, asm

    with open(asm, 'r') as f:
        for line in f:
            if line[:5] not in ['.text', '.code']:
                continue
            xx = line.split()
            for i in xx[1:]:
                if i.lower() in ['=', ';', 'db', 'dd', 'dw', 'dq', 'dt', 'dword'] or i.startswith('__'):
                    break
                if isvalid(i):
                    opcodes.append(i)
                    break
                    # try:
                    #     opcode = line.strip().split()[0]
                    # except IndexError:
                    #     continue
                    # if isvalid(opcode):
                    #     opcodes.append(opcode)
    RESULT[os.path.splitext(asm)[0]] = opcodes

# sys.exit()

with open(os.path.join(opcode_dir, 'opcodes_kaggle.json'), 'w') as f:
    json.dump(RESULT, f)

OPS = set()
os.chdir(opcode_dir)

with open(os.path.join(opcode_dir, 'opcodes_kaggle.json')) as f:
    opcodes = json.load(f)
    opcodes_len = [len(i) for i in opcodes.values()]
    # OPS = [set(i) for i in opcodes.values()]
    OPS = reduce(lambda x, y: x | y, [set(i) for i in opcodes.values()])

    # for i in [set(i) for i in opcodes.values()]:
    #     OPS.update(i)
    # print opcodes_len
    print OPS, len(OPS)
