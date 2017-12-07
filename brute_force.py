#!/usr/bin/python
# coding=utf-8
from kafka import KafkaConsumer
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import MySQLdb

ips = set()
timestep = 10 * 60  # 循环统计时间间隔
timestep_thresh = 60  # 登录间隔时间阈值

ratio = 0.2
login_thresh = 70  # 登录次数阈值
HOST = '10.15.42.21'
DBNAME = 'jsdx'  # 数据库名字，请修改
USER = 'root'  # 数据库账号，请修改
PASSWD = 'root'  # 数据库密码，请修改
PORT = 3306  # 数据库端口，在dbhelper中使用


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
    start_time = 1510588681215
    while 1:
        d = {}  # count, srcs, users, success_sip_user, start_time, end_time, type, ifsuccess, collectTime
        consumer = KafkaConsumer('linux-new3', bootstrap_servers='10.15.42.23:29092', auto_offset_reset='earliest')
        for i, msg in enumerate(consumer):
            data = json.loads(msg.value)
            dst = data['equIP']
            if int(data['collectTime']) < start_time + timestep * 1000:
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

            else:
                # filter that login count above thresh
                # d = {k: v for k, v in d.items() if v[0] > thresh}

                d_flat = map(lambda dst: [dst] + d[dst], d)
                d_flat = map(get_brute_force_type, d_flat)

                d_flat_f = filter(lambda x: x[7], d_flat)
                items = map(lambda x: x[:-1], d_flat_f)

                # break
                conn = MySQLdb.connect(host=HOST, port=PORT, user=USER, passwd=PASSWD, db=DBNAME, charset="utf8")
                cur = conn.cursor()
                cur.executemany('insert into brute_force(dst, count, srcs, users, success_sip_user, '
                                'start_time, end_time, type, ifsuccess) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)', items)
                conn.commit()
                cur.close()
                conn.close()

                d = {}
                start_time += timestep * 1000
                break

        consumer.close()

        break
        time.sleep(timestep)


if __name__ == '__main__':
    detect_brute()
