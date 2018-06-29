# -*- coding: utf-8 -*-

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

log = logging.getLogger(__file__[:-3])
CPU_COUNT = cpu_count()
KEYS = ['2f953c1436cb4803ed92e2897d644f5a3ffeed2d3ee25ff8a99c8ef402414193',
        '7f3b87f683a302eab4c5b49d2765987587aedbd6b76c22c912ebc2af67ff1911',
        'f3cc37e4ca64c62a1df4c5db7c9296dd928a1d7f39c78c88e74829f70e66e15b',
        '863dbdd0b90bf0f2471efeac9bc2dad5e8ee7002b2f6746da6ddd7b8059a1ed3',
        'beb30ccc839cb6c10840b4002263992fe0b3ec1da109c984983f4360283b5764',
        'f677ec5ae0f83ce403b6d340639779581a6f4fa1e273fd9040e9c331a705be1f',
        'e78e0705c8f9e927c2684570bea5782df75ac3c1fd7cc6442d9a1176d9d854c2',
        '20e67d281302544f296f0edf434531ddebec08fa813971624bd46a3876fdbe42',
        'e598222fea48b4e6c6ba3c9fb32376901dad50c3972b5f21b8130d7f705d25f7',
        'a23a8a5c33a5ddd5caad0617fd1c7e18922c910698bf598b666aee167c8f7c08',
        '4a5cd77eb7115589429cb51c1af8fc729b1040225e3fa0415070ce20cd8274bf',
        'f7b31dea8b2661cb69a39669875f3ebfd32d693590095d87bbb0a7cf67c6515a',
        '6468d26f08ce199c189ef57e17c3d3764864999f7ed544e71d34668bd4a89bc9',
        '908e8371f2a843dcccbbd25b249cedf3833380b89b9a71b6b24949aa3ead62ac',
        'f036345feaee8cf4c8ace77bae629288d661b8ec689ac38cf6c969dca6a266ad',
        '73e86529b50ce220b319ffdd2d69995f98d6be5ebb3b4526fe0f7bb30d08e0e7', ]


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


def multi_process(data):
    pool = Pool(processes=CPU_COUNT - 2, initializer=init_worker, maxtasksperchild=400)
    for i in data:
        pool.apply_async(handle_mysql, (i,), callback=handle_mysql)

    pool.close()
    pool.join()


def get_domains(tb_domain='black_domain'):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select domain from {} where category_vt is Null LIMIT 100000'.format(tb_domain)
    cur.execute(sql)
    _r = cur.fetchall()
    # print _r[0]
    cur.close()
    conn.close()
    return _r


def craw_category(domains, tb_domain='black_domain'):
    key_len = len(KEYS)
    url = 'https://www.virustotal.com/vtapi/v2/domain/report'
    # domains = ['fhumellefrictionlessv.com', 'gsvzpartbulkyf.com', '1dndk7boyx065100ahht11qce2d.net',
    #            'xxsncodfxtqsadg.com']
    conn = pymysql.connect(**options)
    cur = conn.cursor()

    for i, _domain in enumerate(domains):
        params = {'apikey': KEYS[i % key_len], 'domain': _domain[0]}
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                _c = str(response.status_code)
            else:
                _r = response.json()
                if 'categories' in _r:
                    _c = _r['categories'][0]
                elif 'Forcepoint ThreatSeeker category' in _r:
                    _c = _r['Forcepoint ThreatSeeker category']
                else:
                    _c = 'uncategorized'
            sql = "update {} set category_vt='{}' where domain='{}'".format(tb_domain, _c, _domain[0])
            log.info(sql)
            cur.execute(sql)
            conn.commit()
        except:
            log.error('insert data failed, msg: %s', traceback.format_exc())

    cur.close()
    conn.close()


if __name__ == '__main__':
    init_logging('/root/chy/{}.log'.format(__file__[:-3]))
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    domains = get_domains()
    # print domains
    craw_category(domains)
