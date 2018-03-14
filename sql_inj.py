#!/usr/bin/python
# coding=utf-8

from kafka import KafkaConsumer
import json
import time
import pymysql
import logging.handlers
import traceback
import requests
import sys
import re
from datetime import datetime

HOST = '10.200.163.6'
DBNAME = 'SSA'
USER = 'root'
PASSWD = '1qazXSW@3edc'
PORT = 3306

sqlmap_server = 'http://127.0.0.1:8775'

kafka_addr = '10.200.163.97:9092'
topic = 'cemb_accesslog'

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


def insert_sql(item):
    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER, passwd=PASSWD, db=DBNAME, charset="utf8")
        cur = conn.cursor()
        cur.execute('insert into ebank_poc_sqlinj(srcIp, time, method, data, url, srcType) values(%s,%s,%s,%s,%s,%s)',
                    item)
        conn.commit()
        cur.close()
        conn.close()
    except:
        log.error('insert data failed, msg: %s', traceback.format_exc())


class AutoSqli(object):
    """
    使用sqlmapapi的方法进行与sqlmapapi建立的server进行交互
    """

    def __init__(self, server='', target='', data='', referer='', cookie=''):
        super(AutoSqli, self).__init__()
        self.server = server
        if self.server[-1] != '/':
            self.server = self.server + '/'
        self.target = target
        self.taskid = ''
        self.engineid = ''
        self.status = ''
        self.data = data
        self.referer = referer
        self.cookie = cookie
        self.start_time = time.time()
        log.debug('Creating an instance of AutoSqli for {0}'.format(self.target))

    def task_new(self):
        try:
            self.taskid = json.loads(
                requests.get(self.server + 'task/new').text)['taskid']
            if len(self.taskid) > 0:
                return True
            return False
        except ConnectionError:
            log.error("sqlmapapi.py is not running")

    def task_delete(self):
        json_kill = requests.get(self.server + 'task/' + self.taskid + '/delete').text
        # if json.loads(requests.get(self.server + 'task/' + self.taskid + '/delete').text)['success']:
        #     #print '[%s] Deleted task' % (self.taskid)
        #     return True
        # return False

    def scan_start(self):
        headers = {'Content-Type': 'application/json'}
        log.info("Starting to scan %s", self.target)
        payload = {'url': self.target}
        url = self.server + 'scan/' + self.taskid + '/start'
        t = json.loads(
            requests.post(url, data=json.dumps(payload), headers=headers).text)
        self.engineid = t['engineid']
        if len(str(self.engineid)) > 0 and t['success']:
            # print 'Started scan'
            return True
        return False

    def scan_status(self):
        self.status = json.loads(
            requests.get(self.server + 'scan/' + self.taskid + '/status').text)['status']
        if self.status == 'running':
            return 'running'
        elif self.status == 'terminated':
            return 'terminated'
        else:
            return 'error'

    def scan_data(self):
        self.data = json.loads(
            requests.get(self.server + 'scan/' + self.taskid + '/data').text)['data']
        if len(self.data) == 0:
            log.info('\033[1;32;40mno injection\033[0m')
            # insert_sql 源IP地址、时间、请求方法/参数
        else:
            # insert_sql(item)
            log.warning('\033[1;33;40minjection\033[0m')

    def option_set(self):
        headers = {'Content-Type': 'application/json'}
        option = {"options": {
            "smart": True,
            "timeout": 60,
        }
        }
        url = self.server + 'option/' + self.taskid + '/set'
        t = json.loads(requests.post(url, data=json.dumps(option), headers=headers).text)

    def scan_stop(self):
        json_stop = requests.get(self.server + 'scan/' + self.taskid + '/stop').text
        # json.loads(
        #     requests.get(self.server + 'scan/' + self.taskid + '/stop').text)['success']

    def scan_kill(self):
        json_kill = requests.get(self.server + 'scan/' + self.taskid + '/kill').text
        # json.loads(
        #     requests.get(self.server + 'scan/' + self.taskid + '/kill').text)['success']

    def run(self):
        if not self.task_new():
            return False
        self.option_set()
        if not self.scan_start():
            return False
        while True:
            if self.scan_status() == 'running':
                time.sleep(1)
            elif self.scan_status() == 'terminated':
                break
            else:
                break
            if time.time() - self.start_time > 60:
                self.scan_stop()
                self.scan_kill()
                break
        self.scan_data()
        self.task_delete()


def detect_sql_inj():
    log.info('start detecting sql injection.')
    s = r"union( |%20|\+|%2B)+select|and( |%20|\+|%2B)+\d=\d|or( |%20|\+|%2B)+\d=\d|select( |%20|\+|%2B).+from( |%20|\+|%2B)|select( |%20|\+|%2B).+version|delete( |%20|\+|%2B).+from( |%20|\+|%2B)|insert( |%20|\+|%2B).+into( |%20|\+|%2B)|update( |%20|\+|%2B).+set( |%20|\+|%2B)|(CREATE|ALTER|DROP|TRUNCATE)( |%20|\+|%2B).+(TABLE|DATABASE)|asc\(|mid\(|char\(|xp_cmdshell|;exec( |%20|\+|%2B)"
    pattern = re.compile(s, re.IGNORECASE)

    # auto_offset_reset should set to 'latest' in real situation, 'earliest' just for test
    try:
        consumer = KafkaConsumer(topic, bootstrap_servers=kafka_addr, auto_offset_reset='latest')
    except:
        log.error('connect kafka failed, msg: %s', traceback.format_exc())
        sys.exit()

    while 1:
        ct = 0
        for ct, msg in enumerate(consumer, 1):
            data = json.loads(msg.value)
            url_all = data['url_s'] + data['data_s']
            ret = re.search(pattern, url_all)
            if ret:
                log.info(data)
                log.warning('\033[1;33;40minjection\033[0m')
                item = [data['srcIp_s'], data['time_dt'], data['method_s'], data['data_s'], data['url_s'], 'httplog']
                insert_sql(item)
            # else:
            #     log.info('\033[1;32;40mno injection\033[0m')
            if ct % 10000 == 0:
                log.info('kafka data count: %s', ct)

        break
    # consumer.close()


if __name__ == '__main__':
    init_logging('sql_inj.log')
    detect_sql_inj()
