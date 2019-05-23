#!/usr/bin/python
# coding=utf-8

import os
import urllib
import urllib2
from time import sleep
import sys
from bs4 import BeautifulSoup
from lxml import etree
import shutil
import requests
import json
from random import choice
import MySQLdb
import logging.handlers
import threading
import Queue
import signal
from multiprocessing import cpu_count, Pool
import argparse
from math import ceil
from functools import partial
from ConfigParser import RawConfigParser
import traceback

reload(sys)
sys.setdefaultencoding('utf8')

failed_page_file = 'tfailed_page.txt'
success_page_file = 'tsuccess_page.txt'
failed_page_file_1 = 'tfailed_page_1.txt'
cnt_file = 'cnt.txt'
ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
cnnvd_url = 'http://www.cnnvd.org.cn/web/vulnerability/querylist.tag'

CPU_COUNT = cpu_count()

log = logging.getLogger(os.path.basename(__file__).split('.')[0])


def init_logging():
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(__file__.split('.')[0] + '.log')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def url_open(url):
    if proxies:
        proxy = choice(proxies)
        handler = urllib2.ProxyHandler(proxy)
        opener = urllib2.build_opener(handler)
        urllib2.install_opener(opener)
    request = urllib2.Request(url, headers={"User-Agent": ua})
    return urllib2.urlopen(request, timeout=10)


def to_page(page, page_file=failed_page_file):
    with open(page_file, 'a') as f:
        f.write(page)
        f.write('\n')


def get_last_page_num(url=cnnvd_url):
    r = url_open(url)
    if r.code == 200:
        data = r.read()
        response = etree.HTML(data)
    else:
        log.error("can't open start url, exit!")
        sys.exit()

    return int(response.xpath(u"//input[@id='pagecount']/@value")[0])


def get_total_count(url=cnnvd_url):
    r = url_open(url)
    if r.code == 200:
        data = r.read()
        response = etree.HTML(data)
    else:
        log.error("can't open start url, exit!")
        sys.exit()

    count_text = response.xpath(u"//a[@onmouse]")[0].text[len(u'总条数：'):]
    count = reduce(lambda x, y: x * 1000 + y, map(lambda x: int(x), count_text.split(',')))
    return count


def get_ldxx(sel):
    if sel.xpath('a'):
        return sel.xpath('a/text()')[0].strip()
    else:
        return ''


def parse_url(url):
    r = url_open(url)
    if r.code == 200:
        data = r.read()
        response = etree.HTML(data)
    else:
        return ''
    sel_ldxxxq = response.xpath('//div[@class="detail_xq w770"]')[0]
    sel_ldxxxq1 = sel_ldxxxq.xpath('ul/li')
    # '漏洞名称 CNNVD编号 危害等级 CVE编号 漏洞类型 发布时间 威胁类型 更新时间 厂商 漏洞来源'
    level, cveid, type, pubtime, ttype, uptime, firm, source = map(get_ldxx, sel_ldxxxq1[1:])
    cname = sel_ldxxxq.xpath('h2/text()')[0]
    cnid = sel_ldxxxq1[0].xpath('span/text()')[0][len(u'CNNVD编号：'):]  # skip prefix[CNNVD编号：]

    brief = ''
    for _b in response.xpath('//div[@class="d_ldjj"]/p'):
        brief += _b.xpath('text()')[0]
    brief = brief.strip()

    _p = response.xpath('//p[@class="ldgg"]')
    patch = _p[0].xpath('text()')[0] if _p else ''
    item = [cnid, cname, pubtime, uptime, level, type, cveid, source, brief, patch, ttype]
    return item


def parse_next_page(next_page):
    conn = MySQLdb.connect(**options)
    cur = conn.cursor()

    r = url_open(next_page)
    if r.code == 200:
        data = r.read()
        response = etree.HTML(data)
        url_pre = 'http://www.cnnvd.org.cn'
        for sel in response.xpath('//div[@class="fl"]/p/a/@href'):
            # remove the situation, ex following
            # http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno=43&repairLd=
            # GraphicsMagick 缓冲区错误漏洞
            if sel.find('CNNVD=CNNVD') != -1:
                url = url_pre + sel
                sleep(0.1)
                item = parse_url(url)
                if item:
                    log.info('cnnvd: %s', item[0])
                    try:
                        cur.execute(
                            'insert into cnnvd(cnid, cname, pubtime, uptime, level, type, cveid, source, '
                            'brief, patch, ttype) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', item)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            log.error('insert_mysql failed! error msg: %s, current url: %s' % (e, url))
                else:
                    log.error("can't open url: %s", url)
        to_page(next_page, success_page_file)

    else:
        to_page(next_page)
        log.error("can't open next_page: %s", next_page)

    cur.close()
    conn.close()


def get_proxy():
    test_url = 'http://www.xunlei.com/'
    request = urllib2.Request("http://www.xicidaili.com/wn", headers={"User-Agent": ua})
    response = urllib2.urlopen(request)
    assert response.code == 200, 'www.xicidaili.com must be available'
    data = response.read()

    root = etree.HTML(data)
    proxies = []
    for tr in root.xpath("//tr[@class]"):
        tr = tr.xpath("td/text()")
        if len(tr) > 2:
            proxy = {'https': 'https://{0}:{1}'.format(tr[0], tr[1])}
            try:
                r = requests.get(test_url, proxies=proxy)
            except:
                pass
            else:
                proxies.append(proxy)
                if len(proxies) >= 10:
                    break
    return proxies


class WorkManager(object):
    def __init__(self, thread_num=10):
        self.work_queue = Queue.Queue()
        self.threads = []
        self.__init_work_queue()
        self.__init_thread_pool(thread_num)

    def __init_thread_pool(self, thread_num):
        for i in range(thread_num):
            self.threads.append(Work(self.work_queue))

    def __init_work_queue(self):
        for i in range(1, last_page_num):
            # 任务入队，Queue内部实现了同步机制
            self.work_queue.put(
                (parse_next_page,
                 "http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno={}&repairLd=".format(i)))

    def wait_allcomplete(self):
        for item in self.threads:
            if item.isAlive():
                item.join()


class Work(threading.Thread):
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.start()

    def run(self):
        # 死循环，从而让创建的线程在一定条件下关闭退出
        while True:
            try:
                do, args = self.work_queue.get(block=True)  # 任务异步出队，Queue内部实现了同步机制
                do(args)
                self.work_queue.task_done()  # 通知系统任务完成
            except:
                log.error(traceback.format_exc())
                to_page(args)
                log.error("can't open next_page: %s", args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", help="crawl all urls", action="store_true", required=False)
    parser.add_argument("-i", "--increment", help="crawl increment urls", action="store_true", required=False)
    parser.add_argument("-f", "--fail", help="crawl failed urls", action="store_true", required=False)
    parser.add_argument("-p", "--proxy", help="crawl proxy address", action="store_true", required=False)
    args = parser.parse_args()
    url_f = "http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno={}&repairLd="

    init_logging()
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    os.chdir(CURRENT_PATH)
    options = read_conf('settings.conf')

    proxies = None
    if args.proxy:
        proxies = get_proxy()
        print 'get_proxy finished!'

    if args.all:
        if not os.path.exists(cnt_file):
            cnt = get_total_count(cnnvd_url)
            with open(cnt_file, 'w') as f:
                f.write(str(cnt))

        latest_page_num = 1
        if os.path.exists(success_page_file):
            latest_page = os.popen('tail -n 1 {}'.format(success_page_file)).read()[:-1]
            latest_page_num = int(latest_page[
                                  len(u'http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno='):
                                  -len(u'&repairLd=')])
        last_page_num = get_last_page_num(cnnvd_url)
        log.info('start from page: %s', latest_page_num)

        work_manager = WorkManager()
        work_manager.wait_allcomplete()

    if args.increment:
        assert os.path.exists(cnt_file), "execute 'python {} -a' first!".format(__file__)
        cnt1_s = os.popen('tail -n 1 {}'.format(cnt_file)).read().strip()
        assert cnt1_s
        cnt1 = int(cnt1_s)
        cnt = get_total_count(cnnvd_url)
        if cnt == cnt1:
            log.info('no more new data!')
            sys.exit()
        with open(cnt_file, 'w') as f:
            f.write(str(cnt))

        delta = int(ceil((cnt - cnt1) / 10.0))
        for i in range(1, delta + 1):
            parse_next_page(url_f.format(i))

    if args.fail:
        if os.path.exists(failed_page_file):
            shutil.move(failed_page_file, failed_page_file_1)
            pool = Pool(processes=5, initializer=init_worker, maxtasksperchild=400)
            with open(failed_page_file_1) as f:
                lines = f.readlines()
            for line in lines:
                pool.apply_async(parse_next_page, (line[:-1],))
            pool.close()
            pool.join()
            if os.path.exists(failed_page_file):
                log.info('failed crawling failed pages!')
            else:
                log.info('finished crawling failed pages!')
