#!/usr/bin/python
# coding=utf-8
from kafka import KafkaConsumer
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

ips = set()
timestep = 600
ratio = 0.2
thresh = 60


def detect_brute():
    while 1:
        d = {}
        consumer = KafkaConsumer('linux-new3', bootstrap_servers='10.15.42.23:29092', auto_offset_reset='earliest')
        for i, msg in enumerate(consumer):
            data = json.loads(msg.value)
            print data
            if data['LoginResult_s'] == 'FAILURE':
                # time_local = time.localtime(float(data['collectTime']) / 1000)
                # dt = time.strftime('%Y-%m-%d %H:%M:%S', time_local)
                # print time_local, '\n', dt

                if data['equIP'] in d:
                    d[data['equIP']]['cnt'] += 1
                    d[data['equIP']]['sip'][data['SourceIP_s']] += 1
                    d[data['equIP']]['user'][data['LoginUser_s']] += 1
                else:
                    d[data['equIP']] = {}
                    d[data['equIP']]['cnt'] = 1
                    d[data['equIP']]['sip'] = {}
                    d[data['equIP']]['user'] = {}
                    d[data['equIP']]['sip'][data['SourceIP_s']] = 1
                    d[data['equIP']]['user'][data['LoginUser_s']] = 1
                # print data['SourceIP_s']
                if i==5:
                    break
            if data['LoginResult_s'] == 'SUCCESS':
                if data['equIP'] in d and data['SourceIP_s'] in d[data['equIP']]['sip']:
                    pass



        for dst in d:
            if d[dst]['cnt'] > thresh:
                print d
        consumer.close()

        break
        time.sleep(10 * 60)


if __name__ == '__main__':
    detect_brute()
