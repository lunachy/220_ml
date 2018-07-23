#!/usr/bin/python
# coding=utf-8
__author = 'chy'

import os
import sys
import time
import signal
import logging.handlers
from ConfigParser import RawConfigParser
import pymysql
from multiprocessing import Pool, cpu_count
import requests
from random import choice
import traceback
import argparse


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def mv_tb():
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select md5 from {} where flag is Null LIMIT 2'.format(tb_file)
    cur.execute(sql)
    cur.close()
    conn.close()

if __name__ == '__main__':
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings_test.conf'))
