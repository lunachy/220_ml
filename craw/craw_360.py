#!/usr/bin/python
# coding=utf-8
__author = 'chy'

import os
import urllib2
import time
import sys
from bs4 import BeautifulSoup
import MySQLdb
import logging.handlers
from ConfigParser import RawConfigParser

reload(sys)
sys.setdefaultencoding('utf8')

ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
head_ua = {'User-Agent': ua}
url360 = 'http://webscan.360.cn/url'
today = time.strftime("%Y-%m-%d")

log = logging.getLogger()


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def read_conf():
    cfg = RawConfigParser()
    cfg.read('conf/settings.conf')
    _keys = ['host', 'user', 'passwd', 'port', 'source_db']
    _options = {_k: cfg.get('mysql', _k).strip() for _k in _keys}
    return _options


def parse(url=url360):
    """parse fileds from 360 url, insert fileds into sql"""
    request = urllib2.Request(url, headers={"User-Agent": ua})
    r = urllib2.urlopen(request)
    if r.code == 200:
        data = r.read()
        soup = BeautifulSoup(data, 'lxml')
        log.info('crawl url[%s]', url)
    else:
        log.info('no data, url[%s]', url)
        return

    # /url/uberry.me.html
    mid = slice(len('/url/'), -len('.html'))
    hrefs = soup.find('div', class_='ld-list-g').find_all('a')
    urls = [a['href'][mid] for a in hrefs]

    conn = MySQLdb.connect(host=options['host'], port=int(options['port']), user=options['user'],
                           passwd=options['passwd'], db=options['source_db'], charset='utf8')
    cur = conn.cursor()
    # items = [[url, today] for url in urls]
    for _url in urls:
        try:
            cur.execute('insert into url360(url, collect_date) values(%s,%s)', [_url, today])
            conn.commit()
        except Exception, e:
            if e.args[0] == 1062:
                pass
            else:
                log.error('insert_mysql failed! error msg: %s, current url: %s' % (e, _url))

    cur.close()
    conn.close()


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_path)
    options = read_conf()
    cfg = RawConfigParser()
    cfg.read('conf/settings.conf')
    logging_dir_path = cfg.get('logging', 'logging_dir_path').strip()
    if os.path.exists(logging_dir_path):
        log_file = os.path.join(logging_dir_path, 'domain', 'url360_' + today + '.log')
    else:
        log_file = os.path.splitext(__file__)[0] + '.log'
    init_logging(log_file)

    parse(url360)
