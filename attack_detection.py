# coding=utf-8
import pymysql
from collections import defaultdict
from datetime import datetime, timedelta

# 密码猜测攻击告警表(PASSWORD_GUESS_ATTACK_ALARM_INFO)
# WEB攻击告警表(WEB_ATTACK_ALARM_INFO)
# 恶意扫描告警表（MALICIOUS_SCAN_ALARM_INFO）
# 恶意程序攻击告警表（MALICIOUS_PROGRAM_ATTACK_ALARM_INFO）
info_tables = ['PASSWORD_GUESS_ATTACK_ALARM_INFO', 'WEB_ATTACK_ALARM_INFO', 'MALICIOUS_SCAN_ALARM_INFO',
               'MALICIOUS_PROGRAM_ATTACK_ALARM_INFO']
predict_tables = ['PASSWORD_GUESS_ATTACK_ALARM_INFO_PERDICT', 'WEB_ATTACK_ALARM_INFO_PERDICT',
                  'MALICIOUS_SCAN_ALARM_INFO_PERDICT', 'MALICIOUS_PROGRAM_ATTACK_ALARM_INFO_PERDICT']


def get_earliest_date(table1):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='', db='', charset='utf8')
    cursor = conn.cursor()
    cursor.execute("SELECT min(ALARM_CREATE_TIME) from {}".format(table1))
    _datetime = cursor.fetchone()

    cursor.close()
    conn.close()
    return _datetime


def get_data(date1, table1):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='', db='', charset='utf8')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DST_ASSET_ID FROM {} WHERE DATE(ALARM_CREATE_TIME)={}".format(table1, date1))
    _dict = defaultdict(lambda: 0)
    for assert_id in cursor.fetchall():
        _dict[assert_id] += 1

    values = map(lambda x: [date1, x, _dict[x]], _dict)

    cursor.executemany(
        'insert into {}(pass_date, assert_id, attack_cnt) values(%s, %s, %s)'.format(table1 + '_PREDICT'), values)
    conn.commit()
    cursor.close()
    conn.close()


today = datetime.now()
for t in info_tables:
    d = datetime.strptime(get_earliest_date(t).split(' ')[0], '%Y-%m-%d')
    while d < today:
        get_data(d, t)
