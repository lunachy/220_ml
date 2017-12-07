#!/usr/bin/python
# coding=utf-8
import os
from time import sleep
import os
import json
import time
import signal
from multiprocessing import cpu_count, Pool

pefile_dir = '/data/root/pe_classify/2017game_test_1'
CPU_COUNT = cpu_count()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def submit(i):
    pefile_path = os.path.join(pefile_dir, i)
    os.system('cuckoo --cwd /data/root/cuckoo submit {}'.format(pefile_path))
    sleep(0.1)


def multi_submit():
    start = time.time()
    pool = Pool(processes=2, initializer=init_worker, maxtasksperchild=400)
    for i in os.listdir(pefile_dir):
        pool.apply_async(submit, (i,))
    pool.close()
    pool.join()
    end = time.time()
    print("multi_submit cost %.2f seconds." % (end - start))


multi_submit()
