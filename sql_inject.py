#!/usr/bin/python
# coding=utf-8

# from kafka import KafkaConsumer
import os
import json
import time
import pymysql
import logging.handlers
import traceback
import requests
import sys
import re
import urllib2
from bs4 import BeautifulSoup
from datetime import datetime
from ConfigParser import RawConfigParser

# sqlmap_server = 'http://127.0.0.1:8775'


log = logging.getLogger('sql_inj')


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def insert_sql(item):
    try:
        conn = pymysql.connect(**options)
        cur = conn.cursor()
        cur.execute('insert into ebank_poc_sqlinj(srcIp, time, method, data, url, srcType) '
                    'values(%s,%s,%s,%s,%s,%s)',
                    item)
        conn.commit()
        cur.close()
        conn.close()
    except:
        log.error('insert data failed, msg: %s', traceback.format_exc())


def craw_url(url):
    def filter_link(_url):
        if _url and _url != '#':
            if _url.startswith('http://www.baidu.com') or _url.startswith('https://www.baidu.com'):
                return True

    ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
    request = urllib2.Request(url, headers={"User-Agent": ua})
    r = urllib2.urlopen(request)
    if r.code == 200:
        data = r.read()
        # log.info(data)
        # response = etree.HTML(data)
        soup = BeautifulSoup(data, 'lxml')
        links = map(lambda link: link.get('href'), soup.find_all('a'))
        return filter(filter_link, links)

    else:
        log.error("can't open start url, exit!")
        sys.exit()


def sqlmap(host):
    urlnew = "http://127.0.0.1:8775/task/new"
    urlscan = "http://127.0.0.1:8775/scan/"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36"}
    pd = requests.get(url=urlnew, headers=headers)
    jsons = pd.json()
    # log.info("New Task, id:%s, success: %s" % (jsons['taskid'], jsons["success"]))
    id = jsons['taskid']
    scan = urlscan + id + "/start"
    data = json.dumps({"url": "{}".format(host)})
    headerss = {"Content-Type": "application/json"}
    scans = requests.post(url=scan, headers=headerss, data=data)
    swq = scans.json()
    log.info('scanid: %s, url:%s, success: %s' % (swq["engineid"], host, swq["success"]))
    status = "http://127.0.0.1:8775/scan/{}/status".format(id)
    thresh = 10
    while thresh > 0:
        staw = requests.get(url=status, headers=headers)
        if staw.json()['status'] == 'terminated':
            datas = requests.get(url='http://127.0.0.1:8775/scan/{}/data'.format(id))
            dat = datas.json()['data']
            log.info('scan result: %s', dat)
            if dat:
                # TODO: modify the inserting operation!
                # insert_sql('xxx')
                pass
            time.sleep(1)
            thresh -= 1
        elif staw.json()['status'] == 'running':
            continue


if __name__ == '__main__':
    init_logging('sql_inj.log')
    log.info('please make sure sqlmap server already started!')
    # log.info('start sqlmapapi server...')
    # sqlmapapi.server()
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    url = 'http://192.168.1.187/DVWA/vulnerabilities/sqli/?id=123&Submit=Submit&user_token=872b69fde2f878dafcdbc1e1c80b91d8'
    urls = craw_url(url)
    # log.info(urls)
    map(sqlmap, urls)
