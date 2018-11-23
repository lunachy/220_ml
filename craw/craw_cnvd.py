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

reload(sys)
sys.setdefaultencoding('utf8')

failed_url_file = 'cnvd_failed_url.txt'
failed_url_file_1 = 'cnvd_failed_url_1.txt'
failed_page_file = 'cnvd_failed_page.txt'
failed_page_file_1 = 'cnvd_failed_page_1.txt'
success_page_file = 'cnvd_success_page.txt'
cnt_file = 'cnvd_cnt.txt'
ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
head_ua = {'User-Agent': ua}
cnvd_url = 'http://www.cnvd.org.cn/flaw/list.htm?flag=true'
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


def to_page(page, page_file):
    with open(page_file, 'a') as f:
        f.write(page)
        f.write('\n')


def url_open(url):
    _cookie = get_cookies()
    _data = ''
    req = requests.session()
    try:
        r = req.get(url, cookies=_cookie, headers=head_ua)  # , proxies=choice(proxies))
        if r.status_code == 200:
            _data = r.content
            # elif r.status_code == 521:
            #     log.info('url[%s] returns 521, second try', url)
            #     _cookie = get_cookies()
            #     r = req.get(url, cookies=_cookie, headers=head_ua)
            #     _data = r.content
    except Exception, e:
        log.error('open url[%s] failed, error msg: %s' % (url, e))

    return _data


def get_total_count(url=cnvd_url):
    data = url_open(url)
    assert data, "can't open start url!"
    soup = BeautifulSoup(data, 'lxml')
    return int(soup.find('div', class_='pages clearfix').find_all('span')[-1].text.split()[1])


def parse_url(url):
    """parse fileds from vul url, insert fileds into sql"""
    _url_pre = 'http://www.cnvd.org.cn'
    data = url_open(url)
    if data:
        soup = BeautifulSoup(data, 'lxml')
        log.info('crawl url[%s]', url)
    else:
        to_page(url, failed_url_file)
        log.info('no data, url[%s]', url)
        return

    try:
        tds = soup.find('table', class_='gg_detail').find_all('td')
        cveid, title, cnvdid, publishtime, cvss, affecttedproduct, description, referencelink, solution, discoverer, \
        vendorpatch, verifymessage, submissiontime, includedtime, updatetime, attachment, type = [''] * 17
        title = soup.h1.string
        for i in range(0, len(tds) - 1, 2):
            k = tds[i].text.strip()
            v = tds[i + 1].text.strip()
            vr = tds[i + 1].text.replace('\t', '').replace('\r\n\r\n', '\r\n').strip()
            # print k, v
            if k == u'CNVD-ID':
                cnvdid = v
            if k == u'公开日期':
                publishtime = v
            if k == u'危害级别':
                cvss = tds[i + 1].text.split('(')[0].strip()
            if k == u'影响产品':
                affecttedproduct = vr
            if k == u'CVE ID':
                cveid = v
            if k == u'漏洞描述':
                description = vr
            if k == u'参考链接':
                referencelink = v
            if k == u'漏洞解决方案':
                solution = vr
            if k == u'厂商补丁':
                if tds[i + 1].find('a'):
                    vendorpatch = v + '\r\n' + _url_pre + tds[i + 1].a['href']
                else:
                    vendorpatch = v
            if k == u'验证信息':
                verifymessage = v
            if k == u'报送时间':
                submissiontime = v
            if k == u'收录时间':
                includedtime = v
            if k == u'更新时间':
                updatetime = v
            if k == u'漏洞附件':
                attachment = v
        item = [cveid, title, cnvdid, publishtime, cvss, affecttedproduct, description, referencelink, solution,
                discoverer, vendorpatch, verifymessage, submissiontime, includedtime, updatetime, attachment, type]
    except Exception, e:
        to_page(url, failed_url_file)
        log.error("parse url[%s] failed, msg: %s." % (url, e))
        return

    conn = MySQLdb.connect(host=options['host'], port=int(options['port']), user=options['user'],
                           passwd=options['passwd'], db=options['source_db'], charset='utf8')
    cur = conn.cursor()
    try:
        cur.execute(
            'insert into cnvd(cveid, title, cnvdid, publishtime, cvss, affecttedproduct, description,'
            'referencelink, solution, discoverer, vendorpatch, verifymessage, submissiontime, includedtime,'
            'updatetime, attachment, type, collect_date)'
            'values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
            item + [today])
        conn.commit()
    except Exception, e:
        if e.args[0] == 1062:
            pass
        else:
            to_page(url, failed_url_file)
            log.error('insert_mysql failed! error msg: %s, current url: %s' % (e, url))
    finally:
        cur.close()
        conn.close()


def parse_next_page(next_page):
    """parse vul urls from page"""
    _url_pre = 'http://www.cnvd.org.cn'
    data = url_open(next_page)
    if data:
        log.info("crawl page[%s]", next_page)
        soup = BeautifulSoup(data, 'lxml')
        for tds in soup.find_all('td', width="45%"):
            url = _url_pre + tds.a['href']
            parse_url(url)
            time.sleep(0.1)
        to_page(next_page, success_page_file)
    else:
        to_page(next_page, failed_page_file)
        log.error("can't open next_page[%s]", next_page)


def get_proxy():
    request = urllib2.Request("http://www.xicidaili.com/wn", headers={"User-Agent": ua})
    response = urllib2.urlopen(request)
    assert response.code == 200, 'www.xicidaili.com must be available'
    data = response.read()

    soup = BeautifulSoup(data, 'lxml')
    proxies = []
    for tr in soup.find_all('tr', class_=True):
        if len(tr) > 2:
            tds = tr.find_all('td')
            proxy = {'https': 'https://{0}:{1}'.format(tds[1].text, tds[2].text)}
            try:
                r = requests.get('http://www.xunlei.com/', proxies=proxy)
            except:
                pass
            else:
                proxies.append(proxy)
                if len(proxies) >= 10:
                    break
    return proxies


def get_cookies(url=cnvd_url):
    req = requests.session()
    r1 = req.get(url, headers=head_ua)

    assert r1.status_code == 521
    js1 = r1.content.strip().replace("<script>", "").replace("</script>", "").replace(
        "eval(y.replace(/\\b\\w+\\b/g, function(y){return x[f(y,z)-1]}))",
        "y.replace(/\\b\\w+\\b/g, function(y){return x[f(y,z)-1]})").replace('\x00', '')

    cookie = r1.cookies.get_dict()

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

    cookie['__jsl_clearance'] = ret2.split('=')[1]
    r2 = req.get(url, cookies=cookie, headers=head_ua)
    return cookie if r2.status_code == 200 else {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", help="crawl all urls", action="store_true", required=False)
    parser.add_argument("-i", "--increment", help="crawl increment urls", action="store_true", required=False)
    parser.add_argument("-f", "--fail", help="crawl failed urls", action="store_true", required=False)
    args = parser.parse_args()
    url_pre = "http://www.cnvd.org.cn/flaw/list.htm?offset="
    root_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_path)

    options = read_conf()
    cfg = RawConfigParser()
    cfg.read('conf/settings.conf')
    logging_dir_path = cfg.get('logging', 'logging_dir_path').strip()
    if os.path.exists(logging_dir_path):
        log_file = os.path.join(logging_dir_path, 'vul', 'cnvd_' + today + '.log')
    else:
        log_file = os.path.splitext(__file__)[0] + '.log'
    init_logging(log_file)

    if args.all:
        last_offset_num = get_total_count()
        if not os.path.exists(cnt_file):
            with open(cnt_file, 'w') as f:
                f.write(str(last_offset_num))

        latest_offset_num = 0
        if os.path.exists(success_page_file):
            latest_page = os.popen('tail -n 1 {}'.format(success_page_file)).read().strip()
            latest_offset_num = int(latest_page[len(url_pre):])
        log.info('start from offset: %s', latest_offset_num)
        log.info('current last offset: %s', last_offset_num)

        for i in range(latest_offset_num, last_offset_num, 20):
            parse_next_page(url_pre + str(i))
            time.sleep(1)

    if args.increment:
        assert os.path.exists(cnt_file), "execute 'python {} -a' first!".format(__file__)
        cnt1_s = os.popen('tail -n 1 {}'.format(cnt_file)).read().strip()
        assert cnt1_s
        cnt1 = int(cnt1_s)
        last_offset_num = get_total_count()
        log.info('crawled count: %s, current total count: %s' % (cnt1, last_offset_num))
        if last_offset_num == cnt1:
            log.info('no more new data!')
            sys.exit()
        with open(cnt_file, 'w') as f:
            f.write(str(last_offset_num))

        delta = last_offset_num - cnt1
        for i in range(0, delta, 20):
            parse_next_page(url_pre + str(i))

    if args.fail:
        # what if it gets stuck, meaning it falls into a dead cycle, it rarely happens however.
        assert not os.path.exists(failed_page_file_1), 'exists {}, handle it first.'.format(failed_page_file_1)
        assert not os.path.exists(failed_url_file_1), 'exists {}, handle it first.'.format(failed_url_file_1)
        while os.path.exists(failed_page_file):
            shutil.move(failed_page_file, failed_page_file_1)
            with open(failed_page_file_1) as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        parse_next_page(line_s)
            time.sleep(60)
        else:
            log.info('no failed page or finished crawling failed pages!')

        while os.path.exists(failed_url_file):
            shutil.move(failed_url_file, failed_url_file_1)
            with open(failed_url_file_1) as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        parse_url(line_s)
            time.sleep(60)
        else:
            log.info('no failed url or finished crawling failed urls!')
