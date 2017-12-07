#!/usr/bin/python
# coding=utf-8
__author = 'chy'

import os
import urllib2
import time
import sys
from bs4 import BeautifulSoup
import shutil
import requests
from random import choice
import MySQLdb
import logging.handlers
import argparse
import PyV8
from ConfigParser import RawConfigParser

failed_url_file = 'cnvd_failed_url.txt'
failed_url_file_10 = 'cnvd_failed_url_10.txt'


def read_conf():
    cfg = RawConfigParser()
    cfg.read('conf/settings.conf')
    _keys = ['host', 'user', 'passwd', 'port', 'source_db']
    _options = {_k: cfg.get('mysql', _k).strip() for _k in _keys}
    return _options


options = read_conf()
conn = MySQLdb.connect(host=options['host'], port=int(options['port']), user=options['user'],
                       passwd=options['passwd'], db=options['source_db'], charset='utf8')
cur = conn.cursor()
with open(failed_url_file_10, 'w')as fw:
    with open(failed_url_file) as f:
        for line in f:
            cnvdid = line.strip().split('/')[-1]
            cur.execute('select 1 from cnvd where cnvdid="{0}"'.format(cnvdid))
            result = cur.fetchone()
            if not result:
                fw.write(line)

cur.close()
conn.close()
