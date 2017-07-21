# coding=utf-8
import os
import time
from multiprocessing import cpu_count, Pool

import pandas as pd
import pefile
import sys
import shutil

root_dir = '/root/pe_classify/'
train_csv = os.path.join(root_dir, '2017game_train.csv')
train_csv_2 = os.path.join(root_dir, '2017game_train_2.csv')
train_pe_dir = os.path.join(root_dir, '2017game_train')
train_pe_dir_1 = os.path.join(root_dir, '2017game_train_1')
train_asm_dir = os.path.join(root_dir, '2017game_train_asm')
train_asm_dir_1 = os.path.join(root_dir, '2017game_train_asm_1')

test_csv = os.path.join(root_dir, '2017game_test.csv')
test_csv_2 = os.path.join(root_dir, '2017game_test_2.csv')
test_pe_dir = os.path.join(root_dir, '2017game_test')
test_pe_dir_1 = os.path.join(root_dir, '2017game_test_1')
test_asm_dir = os.path.join(root_dir, '2017game_test_asm')
test_asm_dir_1 = os.path.join(root_dir, '2017game_test_asm_1')

CPU_COUNT = cpu_count()


def get_pefile(target):
    try:
        pe = pefile.PE(target)
    except:
        return None

    return target


def cp_train_file(target):
    # copy pe file
    shutil.copy2(os.path.join(train_pe_dir, target), train_pe_dir_1)
    # copy asm file
    shutil.copy2(os.path.join(train_asm_dir, target + '.asm'), train_asm_dir_1)


def cp_test_file(target):
    # copy pe file
    shutil.copy2(os.path.join(test_pe_dir, target), test_pe_dir_1)
    # copy asm file
    shutil.copy2(os.path.join(test_asm_dir, target + '.asm'), test_asm_dir_1)


if __name__ == "__main__":
    # # process train dataset
    # os.chdir(train_pe_dir)
    # label = pd.read_csv(train_csv)
    # 
    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, maxtasksperchild=400)
    # pefiles = pool.map(get_pefile, os.listdir(train_pe_dir))
    # pool.close()
    # pool.join()
    # end = time.time()
    # print("cost all time: %s seconds." % (end - start))
    # 
    # pefiles_f = filter(lambda x: x is not None, pefiles)
    # label1 = label[label['md5'].isin(pefiles_f)]
    # c1 = label1.groupby('type').count()
    # types = c1[c1['md5'] >= 100].index.tolist()
    # # ['AdWare', 'Backdoor', 'Banker', 'Clicker', 'Constructor', 'DoS', 'Downloader', 'Dropper', 'Email-Flooder',
    # #  'Email-Worm', 'Exploit', 'Flooder', 'GameThief', 'HackTool', 'Hoax', 'IM', 'IM-Worm', 'Monitor', 'Net-Worm',
    # #  'P2P-Worm', 'PSW', 'Packed', 'Porn-Dialer', 'Proxy', 'Ransom', 'RiskTool', 'Spy', 'Trojan', 'Trojan-Downloader',
    # #  'VirTool', 'Virus', 'WebToolbar', 'Worm']
    # label2 = label1[label1['type'].isin(types)]
    # label2.to_csv(train_csv_2, columns=['md5', 'type'], index=False)
    # 
    # # process test dataset
    # os.chdir(test_pe_dir)
    # label = pd.read_csv(test_csv)
    # 
    # start = time.time()
    # pool = Pool(processes=CPU_COUNT, maxtasksperchild=400)
    # pefiles = pool.map(get_pefile, os.listdir(test_pe_dir))
    # pool.close()
    # pool.join()
    # end = time.time()
    # print("cost all time: %s seconds." % (end - start))
    # 
    # pefiles_f = filter(lambda x: x is not None, pefiles)
    # label1 = label[label['md5'].isin(pefiles_f)]
    # label2 = label1[label1['type'].isin(types)]
    # label2.to_csv(test_csv_2, columns=['md5', 'type'], index=False)

    train_label2 = pd.read_csv(train_csv)
    start = time.time()
    pool = Pool(processes=CPU_COUNT, maxtasksperchild=400)
    for filename in train_label2['md5']:
        pool.apply_async(cp_train_file, (filename,))
    pool.close()
    pool.join()
    end = time.time()
    print("cost all time: %s seconds." % (end - start))

    train_label2 = pd.read_csv(test_csv)
    start = time.time()
    pool = Pool(processes=CPU_COUNT, maxtasksperchild=400)
    for filename in train_label2['md5']:
        pool.apply_async(cp_test_file, (filename,))
    pool.close()
    pool.join()
    end = time.time()
    print("cost all time: %s seconds." % (end - start))
