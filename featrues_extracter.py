# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import json
import pandas as pd
import logging
import MySQLdb
from multiprocessing import cpu_count, Pool
import signal

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='featrues_extracter.log',
                    filemode='w')

total_featrue_path = '/data/root/cuckoo/signatures/windows/'
signatrues_path = '/data/root/cuckoo/storage/analyses/'
pattern = re.compile('    name = "(.*)"')
title_list = []
data_list = []
fail_list = []
title_list = ['md5', 'dumped_buffer', 'dumped_buffer2', 'network_bind']


# /data/root/cuckoo/storage/analyses/1/reports/report.json


def get_md5(i):
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='root123',
        db='sandbox',
    )
    cur = conn.cursor()
    cur.execute("select * from tasks")
    info = cur.fetchall()[i - 1][1]
    md5 = info.split('/')[5]
    return md5


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def extract_total_featrue():
    total_featrue = []
    file_list = os.listdir(total_featrue_path)
    for filename in file_list:
        filename = total_featrue_path + filename
        with open(filename, 'r+') as f:
            for lines in f:
                if not re.findall(pattern, lines) == []:
                    name = re.findall(pattern, lines)
                    # print (name)
                    total_featrue.extend(name)
    return total_featrue


def get_signatrue(file):
    data_series = file
    data = {}
    filename = signatrues_path + file + '/reports/report.json'
    logging.info('{}'.format(filename))
    try:
        with open(filename, 'r') as f:
            j = json.load(f)
            md5 = get_md5(int(file))
            data.update(dict({'md5': md5}))
            if 'signatures' not in j.keys():
                fail_list.append(file)
                logging.warning('{} dont have signatrues'.format(md5))

            else:
                sigs = j['signatures']
                if sigs:
                    for i in xrange(len(sigs)):
                        featrue_name = j['signatures'][i]['name']
                        data.update({featrue_name: 1})
                        data_series = pd.Series(data)
                else:
                    fail_list.append(file)
                    logging.warning('{} {} dont have signatrues'.format(file, md5))
    except Exception as e:
        logging.exception(e)
        return None

    return data_series


total_featrue_list = extract_total_featrue()
# ['trojan_jorik', 'fakeav_mutexes', 'antivm_xen_keys'] duplicate
total_featrue_list = list(set(total_featrue_list))
title_list.extend(total_featrue_list)
label = pd.DataFrame(columns=title_list)
series_type = type(pd.Series())

file_list = os.listdir(signatrues_path)
file_list.remove('latest')
file_list_int = map(int, file_list)
file_list_int_sorted = sorted(file_list_int)
file_list = map(str, file_list_int_sorted)
CPU_COUNT = cpu_count() - 2
pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
data_list = pool.map(get_signatrue, file_list)
pool.close()
pool.join()

data_list_left = data_list
for i, cc in enumerate(data_list_left):
    if type(cc) != series_type:
        fail_list.append(data_list_left[i])
        del data_list_left[i]

label_data = label.append(data_list_left, ignore_index=True)
label_data = label_data.fillna('0')
label_data.index = label_data['md5']
del label_data['md5']
label_data.to_csv('dynamic_featrue.csv')
with open('fail_list.txt', 'w+') as f:
    for i in fail_list:
        f.write(i)
        f.write('\n')
