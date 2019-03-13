# coding=utf-8
# !/usr/bin/env python

import os
from socket import *
import json
import re
from dga_base import run_predict_ml
from ConfigParser import RawConfigParser
import pymysql
import numpy as np


def is_hide_channel(domain):
    if any([not re.search('\w+\.\w+$', domain), domain.isupper()]):
        return True
    else:
        return False


def load_black_domains():
    conn = pymysql.connect(**options['mysql'])
    cur = conn.cursor()
    tb_black_domain = 'black_domain_distinct'
    sql = 'select domain, category_vt from {} where category_vt is not NULL'.format(tb_black_domain)
    cur.execute(sql)
    r = cur.fetchall()
    cur.close()
    conn.close()
    return np.array(r)


def is_malicious(domain):
    if domain in black_domains:
        return True
    else:
        return False


def func_combine(domain):
    funcs = [is_hide_channel, run_predict_ml, is_malicious]
    rets = []
    for func in funcs:
        rets.append(func(domain))
    return rets


def detect_dns():
    s = socket(AF_INET, SOCK_DGRAM)
    s.bind((options['udp']['host'], options['udp']['port']))
    print '...waiting for message..'
    while True:
        data, address = s.recvfrom(4096)
        data = json.loads(data)
        print data, address
        domains = set()
        if 'dns' in data:
            dns_data = data['dns']
            if 'query' in dns_data:
                if 'rrname' in dns_data['query']:
                    domains.add(str(dns_data['query']['rrname']))
            if 'answer' in dns_data:
                if 'rrname' in dns_data['answer']:
                    domains.add(str(dns_data['answer']['rrname']))
                if 'answers' in dns_data['answer']:
                    answers = dns_data['answer']['answers']
                    for ans in answers:
                        if 'rrname' in ans:
                            domains.add(ans['rrname'])
        domains = list(domains)
        results = map(func_combine, domains)
        # TODO: insert into mysql

    # s.close()


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _options = {}
    _dict = {
        'mysql': ['host', 'user', 'passwd', 'port', 'db', 'charset'],
        'udp': ['host', 'port']
    }
    for _k in _dict:
        values = _dict[_k]
        _options[_k] = {}
        for value in values:
            _options[_k].update({value: cfg.get(_k, value).strip()})
    _options['mysql']['port'] = int(_options['mysql']['port'])
    _options['udp']['port'] = int(_options['udp']['port'])
    return _options


if __name__ == '__main__':
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    ret = load_black_domains()
    black_domains = list(ret[:, 0])
    cats = list(ret[:, 1])
    detect_dns()
