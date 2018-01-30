#!/usr/bin/python
# coding=utf-8

from kafka import KafkaConsumer
import json
import time
import MySQLdb
import logging.handlers
import traceback
import requests
from datetime import datetime

HOST = '132.252.12.62'
DBNAME = 'siap'
USER = 'root'
PASSWD = 'qazQAZ123'
PORT = 3306

sqlmap_server = 'http://127.0.0.1:8775'

kafka_addr = '132.224.229.6:9092'
topic = 'linux-login'

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
        conn = MySQLdb.connect(host=HOST, port=PORT, user=USER, passwd=PASSWD, db=DBNAME, charset="utf8")
        cur = conn.cursor()
        cur.execute('insert into sql_inj(dst, count, srcs, users, success_sip_user, '
                    'start_time, end_time, type, ifsuccess) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)', new_item)
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
            # print 'Created new task: ' + self.taskid
            if len(self.taskid) > 0:
                return True
            return False
        except ConnectionError:
            self.logging.error("sqlmapapi.py is not running")

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
        else:
            # insert_sql(item)
            log.warning('\033[1;33;40minjection\033[0m')

    def option_set(self):
        headers = {'Content-Type': 'application/json'}
        option = {"options": {
            "randomAgent": True,
            "tech": "BT"
        }
        }
        url = self.server + 'option/' + self.taskid + '/set'
        t = json.loads(requests.post(url, data=json.dumps(option), headers=headers).text)
        # print t

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
            if time.time() - self.start_time > 100:
                self.scan_stop()
                self.scan_kill()
                break
        self.scan_data()
        self.task_delete()


def detect_sql_inj():
    log.info('start detecting sql injection.')
    # auto_offset_reset should set to 'latest' in real situation, 'earliest' just for test
    # try:
    #     consumer = KafkaConsumer(topic, bootstrap_servers=kafka_addr, auto_offset_reset='latest')
    # except:
    #     log.error('connect kafka failed, msg: %s', traceback.format_exc())
    #     sys.exit()

    target_file = '/home/ml/chy/sqlmapapi_pi/data/targets.txt'
    with open(target_file) as f:
        data = f.readlines()
    while 1:
        ct = 0
        for ct, msg in enumerate(data, 1):
            # data = json.loads(msg.value)
            # url = data['url']
            url = msg.strip()
            t = AutoSqli(sqlmap_server, url)
            t.run()

        log.info('kafka data count: %s', ct)
        break

    # consumer.close()


if __name__ == '__main__':
    init_logging('/data/log/sql_inj.log')
    detect_sql_inj()
