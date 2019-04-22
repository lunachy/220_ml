# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import logging.handlers
from ConfigParser import RawConfigParser
import pymysql
from random import choice
from datetime import timedelta, datetime


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
    # init_logging('/root/chy/{}.log'.format(__file__[:-3]))
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    _types = ['ip', 'domain', 'url']
    _tbs = list(map(lambda x: 'black_' + x, _types))
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    options['db'] = 'SSA'
    conn1 = pymysql.connect(**options)
    cur1 = conn1.cursor()
    for _type, _tb in zip(_types, _tbs):
        select_sql = 'select {},signature,score,source,collect_date from {}'.format(_type, _tb)
        cur.execute(select_sql)
        _rs = cur.fetchall()
        for _r in _rs:
            last_seen = _r[-1] + timedelta(days=choice(range(1, 7)))
            _level = round(float(_r[2]) / 20.0) if _r[2] else 0
            _keys = '{},`type`,score,source,collect_date,first_seen,last_seen,update_date,`level`'.format(_type)
            _values = _r + [_r[-1], last_seen, last_seen, _level]
            insert_sql = 'insert into {}({}) values({})'.format(_tb, _keys, ','.join(['%s'] * len(_keys.split(','))))
            cur1.execute(insert_sql, _values)
            conn1.commit()
            sys.exit()
    cur.close()
    conn.close()
    cur1.close()
    conn1.close()
