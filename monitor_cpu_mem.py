# coding=utf-8
import time
import psutil
import os
import pymysql
from ConfigParser import RawConfigParser


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def monitor_cpu_mem(time_delta=5):
    table = 'system_status'
    keys = ['cpu_percent', 'mem_percent']
    while True:
        cpu_percent = psutil.cpu_percent()
        mem_percent = psutil.virtual_memory().percent
        conn = pymysql.connect(**options)
        cur = conn.cursor()
        columns = ','.join(keys)
        value = [cpu_percent, mem_percent]
        sql = 'insert into {}({}) values({})'.format(table, columns, ','.join(['%s'] * len(value)))
        cur.execute(sql, value)
        conn.commit()

        cur.close()
        conn.close()

        time.sleep(time_delta)


if __name__ == '__main__':
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    monitor_cpu_mem()
