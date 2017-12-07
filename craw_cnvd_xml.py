#!/usr/bin/python
# coding=utf-8

import os
import urllib
import urllib2
import time
import sys
from bs4 import BeautifulSoup
from lxml import etree
import shutil
import requests
import json
from random import randint
import MySQLdb
import logging.handlers
import threading
import Queue
import signal
from multiprocessing import cpu_count, Pool
import argparse
from math import ceil
import PyV8
from xml.dom.minidom import parse
import xml.dom.minidom

reload(sys)
sys.setdefaultencoding('utf8')

failed_page_file = 'cnvd_failed_page.txt'
success_page_file = 'cnvd_success_page.txt'
failed_page_file_1 = 'cnvd_failed_page_1.txt'
cnt_file = 'cnvd_cnt.txt'
ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
head_ua = {'User-Agent': ua}
cnvd_url = 'http://www.cnvd.org.cn/shareData/list'
log_file = os.path.splitext(__file__)[0] + '.log'

log = logging.getLogger()
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

fh = logging.handlers.WatchedFileHandler(log_file)
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)

HOST = '10.21.37.198'
DBNAME = 'CTI_SOURCE'  # 数据库名字，请修改
USER = 'ml'  # 数据库账号，请修改
PASSWD = '123456'  # 数据库密码，请修改
PORT = 3306  # 数据库端口，在dbhelper中使用
CPU_COUNT = cpu_count()
today = time.strftime("%Y-%m-%d")
cookies = {}


def to_page(page, page_file=failed_page_file):
    with open(page_file, 'a') as f:
        f.write(page)
        f.write('\n')


def url_open(url):
    global cookies
    _data = ''
    req = requests.session()
    try:
        r = req.get(url, cookies=cookies, headers=head_ua)
        if r.status_code == 200:
            _data = r.content
        elif r.status_code == 521:
            cookies = get_cookies()
            r = req.get(url, cookies=cookies, headers=head_ua)
            _data = r.content
    except Exception, e:
        log.error('open url[%s] failed, error msg: %s' % (url, e))

    return _data


def get_total_count(url='http://www.cnvd.org.cn/shareData/list?max=10&offset=0'):
    data = url_open(url)
    with open('/tmp/share.html', 'w') as f:
        f.write(data)
    assert data, "can't open start url!"
    response = etree.HTML(data)
    print response.xpath(u"//div[@class='pages clearfix']/span")
    return int(response.xpath(u"//div[@class='pages clearfix']/span")[-1].text.split()[1])


def get_tagname(node, tagname):
    tagNode = node.getElementsByTagName(tagname)
    return tagNode[0].childNodes[0].data if tagNode else ''


def parse_url(url):
    data = url_open(url)
    if data:
        response = etree.HTML(data)
    else:
        log.error('no data, url[%s]', url)
        return

    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parseString(data)
    collection = DOMTree.documentElement
    items = []
    vuls = collection.getElementsByTagName('vulnerability')
    log.info('vuls count: %s, url[%s]' % (len(vuls), url))
    for vul in vuls:
        cveid = '\r\n'.join(map(lambda x: x.childNodes[0].data, vul.getElementsByTagName('cveNumber')))
        title = get_tagname(vul, 'title')
        cnvdid = get_tagname(vul, 'number')
        publishtime = get_tagname(vul, 'openTime')
        cvss = get_tagname(vul, 'serverity')
        affecttedproduct = '\r\n'.join(map(lambda x: x.childNodes[0].data.replace('&gt;', '>').replace('&lt;', '<'),
                                           vul.getElementsByTagName('product')))
        description = get_tagname(vul, 'description')
        referencelink = get_tagname(vul, 'referenceLink')
        solution = get_tagname(vul, 'formalWay')
        discoverer = get_tagname(vul, 'discovererName')
        vendorpatch = get_tagname(vul, 'patchName')
        verifymessage = ''
        submissiontime = get_tagname(vul, 'submitTime')
        includedtime = get_tagname(vul, 'openTime')
        updatetime = get_tagname(vul, 'openTime')
        attachment = ''
        type = get_tagname(vul, 'isEvent')
        items.append([cveid, title, cnvdid, publishtime, cvss, affecttedproduct, description, referencelink, solution,
                      discoverer, vendorpatch, verifymessage, submissiontime, includedtime, updatetime, attachment,
                      type, today])

    conn = MySQLdb.connect(host=HOST, port=PORT, user=USER, passwd=PASSWD, db=DBNAME, charset="utf8")
    cur = conn.cursor()
    try:
        cur.executemany(
            'insert into cnvd(cveid, title, cnvdid, publishtime, cvss, affecttedproduct, description, referencelink, '
            'solution, discoverer, vendorpatch, verifymessage, submissiontime, includedtime, updatetime, attachment, '
            'type, collect_date) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', items)
        conn.commit()
    except Exception, e:
        if e.args[0] == 1062:
            pass
        else:
            log.error('insert data to sql failed, url[%s] msg: %s' % (url, e))
    cur.close()
    conn.close()


def parse_next_page(next_page):
    data = url_open(next_page)
    if data:
        response = etree.HTML(data)
        _url_pre = 'http://www.cnvd.org.cn'
        for sel in response.xpath('//td[@width="65%"]/a/@href'):
            url = _url_pre + sel
            time.sleep(0.1)
            parse_url(url)

        to_page(next_page, success_page_file)

    else:
        to_page(next_page)
        log.error("can't open next_page: %s", next_page)


def get_cookies(url=cnvd_url):
    req = requests.session()
    r1 = req.get(url, headers=head_ua)

    assert r1.status_code == 521
    js1 = r1.content.strip().replace("<script>", "").replace("</script>", "").replace(
        "eval(y.replace(/\\b\\w+\\b/g, function(y){return x[f(y,z)-1]}))",
        "y.replace(/\\b\\w+\\b/g, function(y){return x[f(y,z)-1]})").replace('\x00', '')

    cookies = r1.cookies.get_dict()

    ctxt = PyV8.JSContext()
    ctxt.enter()
    ret1 = ctxt.eval(js1)
    try:
        js2 = ret1[ret1.index('var cd'): ret1.index('setTimeout')] + ' dc;'
        # consider the browser kernel situation
        if js2.find('document.createElement') != -1:
            js2 = js2[:js2.index('document.createElement')] + "'www.cnvd.org.cn/'" + js2[js2.index(';return function'):]
    except Exception, e:
        log.error('get js failed, error msg: %s', e)
        sys.exit()

    ret2 = ctxt.eval(js2)

    cookies['__jsl_clearance'] = ret2.split('=')[1]
    r2 = req.get(url, cookies=cookies, headers=head_ua)
    if r2.status_code == 200:
        return cookies
    else:
        return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", help="crawl all urls", action="store_true", required=False)
    parser.add_argument("-i", "--increment", help="crawl increment urls", action="store_true", required=False)
    parser.add_argument("-f", "--fail", help="crawl failed urls", action="store_true", required=False)
    args = parser.parse_args()
    url_pre = "http://www.cnvd.org.cn/shareData/list?max=10&offset="
    root_path = '/home/ml/chy'
    os.chdir(root_path)

    cookies = get_cookies()
    assert cookies
    #
    # parse_url('http://www.cnvd.org.cn/shareData/download/395')
    # print get_total_count()

    if args.all:
        # last_offset_num = get_total_count(cnvd_url)
        # if not os.path.exists(cnt_file):
        #     with open(cnt_file, 'w') as f:
        #         f.write(str(last_offset_num))
        #
        # latest_offset_num = 0
        # if os.path.exists(success_page_file):
        #     latest_page = os.popen('tail -n 1 {}'.format(success_page_file)).read().strip()
        #     latest_offset_num = int(latest_page[len(url_pre):])
        # print latest_offset_num
        # log.info('start from offset: %s', latest_offset_num)
        # log.info('current last offset: %s', last_offset_num)

        start_page, end_page = 243, 395
        download_url_pre = 'http://www.cnvd.org.cn/shareData/download/'
        for i in range(start_page, end_page + 1):
            parse_url(download_url_pre + str(i))
            time.sleep(0.1)
            # break

    if args.increment:
        assert os.path.exists(cnt_file), "execute 'python {} -a' first!".format(__file__)
        cnt1_s = os.popen('tail -n 1 {}'.format(cnt_file)).read().strip()
        assert cnt1_s
        cnt1 = int(cnt1_s)
        last_offset_num = get_total_count(cnvd_url)
        log.info('crawled count: %s, current total count: %s' % (cnt1, last_offset_num))
        if last_offset_num == cnt1:
            log.info('no more new data!')
            sys.exit()
        with open(cnt_file, 'w') as f:
            f.write(str(last_offset_num))

        delta = int(ceil((last_offset_num - cnt1) / 10.0))
        for i in range(1, delta + 1):
            parse_next_page(url_pre + str(i))

    if args.fail:
        if os.path.exists(failed_page_file):
            shutil.move(failed_page_file, failed_page_file_1)
            with open(failed_page_file_1) as f:
                for line in f:
                    if line.strip():
                        parse_next_page(line.strip())
            if os.path.exists(failed_page_file):
                log.info('failed crawling failed pages!')
            else:
                log.info('finished crawling failed pages!')
