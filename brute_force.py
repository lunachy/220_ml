#!/usr/bin/python
# coding=utf-8
import sys

sys.path.append('/data1/aus/packages')
from kafka import KafkaConsumer
import json
import time
import MySQLdb
import logging.handlers
import traceback

ips = set()
timestep = 10 * 60  # 循环统计时间间隔
timestep_thresh = 60  # 登录间隔时间阈值

ratio = 0.2
login_thresh = 70  # 登录次数阈值
HOST = '132.252.12.62'
DBNAME = 'jsdx'
USER = 'root'
PASSWD = 'qazQAZ123'
PORT = 3306

kafka_addr = '132.224.229.6:9092'
topic = 'linux-login'

log = logging.getLogger('brute_force')


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def get_timedelta(timestamps):
    ts = map(lambda x: int(x) / 1000, timestamps)
    ds = []
    for i in range(len(ts) - 1):
        ds.append(ts[i + 1] - ts[i])
    return ds


def get_brute_force_type(d_flat):
    ts = map(lambda x: int(x) / 1000, d_flat[-1])
    ds = []
    d_flat[7] = ''
    for i in range(len(ts) - 1):
        ds.append(ts[i + 1] - ts[i])
        if ds[-1] > timestep_thresh:
            pass
        if len(ds) >= 5 and sum(ds[-5:]) <= 1:
            d_flat[7] = u'快速频繁登录'
        if len(ds) >= 10 and ds[-10:] in [[i] * 10 for i in range(1, 4)]:
            d_flat[7] = u'同频率多次登录'
        if len(ds) >= login_thresh:
            d_flat[7] = u'登录次数过多'
    return d_flat


def detect_brute():
    log.info('start detecting brute force.')
    while 1:
        d = {}  # count, srcs, users, success_sip_user, start_time, end_time, type, ifsuccess, collectTime
        # auto_offset_reset should set to 'latest' in real situation, 'earliest' just for test
        try:
            consumer = KafkaConsumer(topic, bootstrap_servers=kafka_addr, auto_offset_reset='latest')
        except:
            log.error('connect kafka failed, msg: %s', traceback.format_exc())
            sys.exit()

        for ct, msg in enumerate(consumer, 1):
            # if ct > 2500: break  # just for test, about 2500 data records in 10min
            data = json.loads(msg.value)
            dst = data['equIP']
            if data['LoginResult_s'] == 'FAILURE':
                time_local = time.localtime(float(data['collectTime']) / 1000)
                dt = time.strftime('%Y-%m-%d %H:%M:%S', time_local)
                # print time_local, '\n', dt

                if dst not in d:
                    d[dst] = [1, {data['SourceIP_s']}, {data['LoginUser_s']}, set(), dt, '', '', 0,
                              [data['collectTime']]]
                else:
                    d[dst][0] += 1
                    d[dst][1].add(data['SourceIP_s'])
                    d[dst][2].add(data['LoginUser_s'])
                    d[dst][5] = dt
                    d[dst][-1].append(data['collectTime'])

            if data['LoginResult_s'] == 'SUCCESS':
                if dst in d:
                    if data['SourceIP_s'] in d[dst][1] and data['LoginUser_s'] in d[dst][2]:
                        d[dst][7] = 1
                        d[dst][3].add(','.join([data['SourceIP_s'], data['LoginUser_s']]))
        log.info('kafka data count: %s', ct)
        consumer.close()

        d_flat = map(lambda dst: [dst] + d[dst], d)
        d_flat = map(get_brute_force_type, d_flat)

        d_flat_f = filter(lambda x: x[7], d_flat)
        items = map(lambda x: x[:-1], d_flat_f)

        try:
            conn = MySQLdb.connect(host=HOST, port=PORT, user=USER, passwd=PASSWD, db=DBNAME, charset="utf8")
            cur = conn.cursor()
            cur.executemany('insert into brute_force(dst, count, srcs, users, success_sip_user, '
                            'start_time, end_time, type, ifsuccess) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)', items)
            conn.commit()
            cur.close()
            conn.close()
        except:
            log.error('connect kafka failed, msg: '
                      '%s', traceback.format_exc())
        else:
            log.info('insert data to mysql, count: %s', len(items))

        time.sleep(timestep)


if __name__ == '__main__':
    init_logging('/data1/aus/brute_force.log')
    detect_brute()
