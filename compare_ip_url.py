# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import socket
import struct
import logging.handlers
from ConfigParser import RawConfigParser
from multiprocessing import Pool, cpu_count
import pymysql
from functools import partial

log = logging.getLogger(__file__[:-3])


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def get_asset_from_tb(tb_asset='am_asset'):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select asset_id,assert_out_ip,url from {}'.format(tb_asset)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def compare(data, tb_black_ip='black_ip', tb_black_url='black_url'):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    for _asset_id, _ip, _url in data:
        if _ip:
            sql = 'select ip from {} where ip="{}"'.format(tb_black_ip, _ip)
            cur.execute(sql)
            _r = cur.fetchone()
            if _r:
                print _asset_id, _r[0]
        if _url:
            sql = 'select url from {} where url="{}"'.format(tb_black_url, _url)
            cur.execute(sql)
            _r = cur.fetchone()
            if _r:
                print _asset_id, _r[0]
    cur.close()
    conn.close()


def is_between_ip(data1, ip_ranges):
    _ip = data1[1]
    if _ip:
        _ip_int = socket.ntohl(struct.unpack("I", socket.inet_aton(_ip))[0])
        for _ip_range in ip_ranges:
            if _ip_range[0] <= _ip_int <= _ip_range[1]:
                return 1
    return 0


def search_ip(data, key_location=r'吉林省', isp=r'中国移动', tb_ip_data='ip_data'):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select min_ip,max_ip from {} where province="{}" and ISP="{}"'.format(tb_ip_data, key_location, isp)
    cur.execute(sql)
    _r = cur.fetchall()
    if _r:
        print len(_r)

    is_between_ip_p = partial(is_between_ip, ip_ranges=_r)
    data_f = filter(is_between_ip_p, data)
    for _data in data_f:
        print _data[0], _data[1]
    cur.close()
    conn.close()


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


if __name__ == '__main__':
    init_logging('/root/chy/compare_ip_url.log')
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    results = get_asset_from_tb()
    compare(results)
    search_ip(results)
