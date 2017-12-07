#!/usr/bin/python
# coding=utf-8
__author = 'chy'

import os
import urllib2
import time
import sys
from bs4 import BeautifulSoup
from lxml import etree
import shutil
import requests
import MySQLdb
import logging.handlers
import argparse
from math import ceil
from ConfigParser import RawConfigParser

reload(sys)
sys.setdefaultencoding('utf8')

failed_url_file = 'cnnvd_failed_url.txt'
failed_url_file_1 = 'cnnvd_failed_url_1.txt'
failed_page_file = 'cnnvd_failed_page.txt'
failed_page_file_1 = 'cnnvd_failed_page_1.txt'
success_page_file = 'cnnvd_success_page.txt'
cnt_file = 'cnnvd_cnt.txt'
ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
cnnvd_url = 'http://www.cnnvd.org.cn/web/vulnerability/querylist.tag'
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


def get_last_page_num(url=cnnvd_url):
    request = urllib2.Request(url, headers={"User-Agent": ua})
    r = urllib2.urlopen(request)
    if r.code == 200:
        data = r.read()
        response = etree.HTML(data)
    else:
        log.error("can't open start url, exit!")
        sys.exit()

    return int(response.xpath(u"//input[@id='pagecount']/@value")[0])


def get_total_count(url=cnnvd_url):
    request = urllib2.Request(url, headers={"User-Agent": ua})
    r = urllib2.urlopen(request)
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
    ldxx = ''
    if sel.xpath('a'):
        try:
            ldxx = sel.xpath('a/text()')[0].strip()
        except:
            pass
    return ldxx


def parse_url(url):
    """parse fileds from vul url, insert fileds into sql"""
    request = urllib2.Request(url, headers={"User-Agent": ua})
    r = urllib2.urlopen(request)
    if r.code == 200:
        try:
            data = r.read()
        except Exception, e:
            # Incomplete url: http://www.cnnvd.org.cn/web/xxk/ldxqById.tag?CNNVD=CNNVD-200508-076
            # Error: IncompleteRead(3788 bytes read, 142 more expected)
            log.error('read data failed, url[%s], msg: %s' % (url, e))
            return
        else:
            response = etree.HTML(data)
            log.info('crawl url[%s]', url)
    else:
        to_page(url, failed_url_file)
        log.info('no data, url[%s]', url)
        return

    try:
        sel_ldxxxq = response.xpath('//div[@class="detail_xq w770"]')[0]
        sel_ldxxxq1 = sel_ldxxxq.xpath('ul/li')
        # '漏洞名称 CNNVD编号 危害等级 CVE编号 漏洞类型 发布时间 威胁类型 更新时间 厂商 漏洞来源'
        level, cveid, type, pubtime, ttype, uptime, manufacturer, source = map(get_ldxx, sel_ldxxxq1[1:])
        cname = sel_ldxxxq.xpath('h2/text()')[0]
        cnid = sel_ldxxxq1[0].xpath('span/text()')[0][len(u'CNNVD编号：'):]  # skip prefix[CNNVD编号：]

        brief = ''
        for _b in response.xpath('//div[@class="d_ldjj"]/p'):
            brief += _b.xpath('text()')[0]
        brief = brief.strip()

        patch = ''
        _p = response.xpath('//p[@class="ldgg"]')
        if _p and _p[0].xpath('text()'):
            patch = _p[0].xpath('text()')[0].strip()

        affecttedproduct = ''
        aps = response.xpath('//div[@class="vulnerability_list"]/ul')[0]
        apsa = aps.xpath('li/div/a')
        if apsa:
            affecttedproduct = '\r\n'.join(map(lambda ap: ap.text.strip(), apsa))

        item = [cnid, cname, pubtime, uptime, manufacturer, level, type, cveid, source, affecttedproduct, brief, patch,
                ttype]
    except Exception, e:
        to_page(url, failed_url_file)
        log.error("parse url[%s] failed, msg: %s." % (url, e))
        return

    conn = MySQLdb.connect(host=options['host'], port=int(options['port']), user=options['user'],
                           passwd=options['passwd'], db=options['source_db'], charset='utf8')
    cur = conn.cursor()

    try:
        cur.execute(
            'insert into cnnvd_copy1(cnid, cname, pubtime, uptime, manufacturer, level, type, cveid, source, '
            'affecttedproduct, brief, patch, ttype, collect_date) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
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
    request = urllib2.Request(next_page, headers={"User-Agent": ua})
    r = urllib2.urlopen(request)
    if r.code == 200:
        log.info("crawl page[%s]", next_page)
        data = r.read()
        response = etree.HTML(data)
        url_pre = 'http://www.cnnvd.org.cn'
        for sel in response.xpath('//div[@class="fl"]/p/a/@href'):
            # remove the situation, ex following
            # http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno=43&repairLd=
            # GraphicsMagick 缓冲区错误漏洞
            if sel.find('CNNVD=CNNVD') != -1:
                url = url_pre + sel
                parse_url(url)
            time.sleep(0.1)
        to_page(next_page, success_page_file)
    else:
        to_page(next_page, failed_page_file)
        log.error("can't open next_page: %s", next_page)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", help="crawl all urls", action="store_true", required=False)
    parser.add_argument("-i", "--increment", help="crawl increment urls", action="store_true", required=False)
    parser.add_argument("-f", "--fail", help="crawl failed urls", action="store_true", required=False)
    parser.add_argument("-p", "--proxy", help="crawl proxy address", action="store_true", required=False)
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true", required=False)
    args = parser.parse_args()
    url_f = "http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno={}&repairLd="
    root_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_path)

    options = read_conf()
    cfg = RawConfigParser()
    cfg.read('conf/settings.conf')
    logging_dir_path = cfg.get('logging', 'logging_dir_path').strip()
    if os.path.exists(logging_dir_path):
        log_file = os.path.join(logging_dir_path, 'vul', 'cnnvd_' + today + '.log')
    else:
        log_file = os.path.splitext(__file__)[0] + '.log'
    init_logging(log_file)

    if args.all:
        if not os.path.exists(cnt_file):
            cnt = get_total_count(cnnvd_url)
            with open(cnt_file, 'w') as f:
                f.write(str(cnt))

        latest_page_num = 1
        if os.path.exists(success_page_file):
            latest_page = os.popen('tail -n 1 {}'.format(success_page_file)).read().strip()
            latest_page_num = int(latest_page[
                                  len(u'http://www.cnnvd.org.cn/web/vulnerability/querylist.tag?pageno='):
                                  -len(u'&repairLd=')])
        last_page_num = get_last_page_num(cnnvd_url)
        log.info('start from page: %s', latest_page_num)
        log.info('current last page: %s', last_page_num)

        for i in range(latest_page_num, last_page_num + 1):
            parse_next_page(url_f.format(i))

        last_page_num_1 = get_last_page_num(url_f.format(i))
        if last_page_num < last_page_num_1:
            for i in range(last_page_num + 1, last_page_num_1 + 1):
                parse_next_page(url_f.format(i))

    if args.increment:
        assert os.path.exists(cnt_file), "execute 'python {} -a' first!".format(__file__)
        cnt1_s = os.popen('tail -n 1 {}'.format(cnt_file)).read().strip()
        assert cnt1_s
        cnt1 = int(cnt1_s)
        last_offset_num = get_total_count(cnnvd_url)
        log.info('crawled count: %s, current total count: %s' % (cnt1, last_offset_num))
        if last_offset_num == cnt1:
            log.info('no more new data!')
            sys.exit()
        with open(cnt_file, 'w') as f:
            f.write(str(last_offset_num))

        delta = int(ceil((last_offset_num - cnt1) / 10.0))
        for i in range(1, delta + 1):
            parse_next_page(url_f.format(i))

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
            time.sleep(10)
        else:
            log.info('no failed page or finished crawling failed pages!')

        while os.path.exists(failed_url_file):
            shutil.move(failed_url_file, failed_url_file_1)
            with open(failed_url_file_1) as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        parse_url(line_s)
            time.sleep(10)
        else:
            log.info('no failed url or finished crawling failed urls!')

    if args.proxy:
        proxies = get_proxy()
        print get_total_count(cnnvd_url)
        print get_last_page_num(cnnvd_url)
