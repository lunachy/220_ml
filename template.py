# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import logging.handlers
from ConfigParser import RawConfigParser
import pymysql
from multiprocessing import Pool, cpu_count

log = logging.getLogger(__file__[:-3])
CPU_COUNT = cpu_count()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def handle_mysql(tb=''):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select * from {}'.format(tb)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def func(para1):
    pass


def callback_func(para1):
    pass


def multi_process(data):
    pool = Pool(processes=CPU_COUNT - 2, initializer=init_worker, maxtasksperchild=400)
    for i in data:
        pool.apply_async(func, (i,), callback=callback_func)

    pool.close()
    pool.join()


class DB(object):
    def __init__(self):
        self.conn = pymysql.connect(**options)
        self.cur = self.conn.cursor()
        # self.num = self.cos.execute()

    def __enter__(self):
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()
        self.cur.close()


with DB() as cur:
    sql = 'xxx'
    cur.execute(sql)
    result = cur.fetchall()
    
if __name__ == '__main__':
    init_logging('/root/chy/{}.log'.format(__file__[:-3]))
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
