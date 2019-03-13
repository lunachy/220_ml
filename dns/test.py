# -*- coding: utf-8 -*-
import sys
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(CURRENT_PATH, 'packages_centos7'))
from collections import Counter
import json
import math
import traceback
import uuid
import logging
import ConfigParser
from ConfigParser import RawConfigParser
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from kafka import KafkaConsumer
import tldextract
import pymysql

CONFIG_PARSER = ConfigParser.ConfigParser()
CONFIG_PARSER.read(CURRENT_PATH + "/settings_dga.conf")
HOST = CONFIG_PARSER.get("mysql", "host")
PORT = CONFIG_PARSER.get("mysql", "port")
USER = CONFIG_PARSER.get("mysql", "user")
PASSWORD = CONFIG_PARSER.get("mysql", "password")
DB = CONFIG_PARSER.get("mysql", "db")
LOGFILENAME = CURRENT_PATH + '/some.log'
KAFKA_ADDR = CONFIG_PARSER.get("kafka", "kafka_addr")
TOPIC = CONFIG_PARSER.get("kafka", "topic")
TB_BLACK_DOMAIN = CONFIG_PARSER.get("mysql_threatinfo", "tb_black_domain")
ALARM_HOSTOUTREACH = CONFIG_PARSER.get("mysql", "alarm_hostoutreach")
ALARM_DGA_TABLE = CONFIG_PARSER.get("mysql", "alarm_dga")
ALARM_REF_TABLE = CONFIG_PARSER.get("mysql", "alarm_ref_event")
ALARM_INFO_TABLE = CONFIG_PARSER.get("mysql", "alarm_info")
KAFKA_FLAG = eval(CONFIG_PARSER.get("conf", "kafka_flag"))
DOMAIN_FLAG = eval(CONFIG_PARSER.get("conf", "domain_flag"))


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _options = {}
    _dict = {
        'mysql_threatinfo': ['host', 'user', 'passwd', 'port', 'db', 'charset'],
    }
    for _k in _dict:
        values = _dict[_k]
        _options[_k] = {}
        for value in values:
            _options[_k].update({value: cfg.get(_k, value).strip()})
    _options['mysql_threatinfo']['port'] = int(_options['mysql_threatinfo']['port'])
    return _options


def load_black_domains():
    conn = pymysql.connect(**options['mysql_threatinfo'])
    cur = conn.cursor()
    sql = 'select domain, level, type from {} where category_vt is not NULL AND category_vt != "uncategorized"'.format(
        TB_BLACK_DOMAIN)
    cur.execute(sql)
    r = cur.fetchall()
    cur.close()
    conn.close()
    return np.array(r)


def run_domain_hostoutreach(domain):
    if domain in black_domains:
        return 1
    else:
        return 0



def insert_mysql(alarm_table, org_id, event_id, domain, src_ip, dst_ip, connect_time, src_country,
                 src_province, src_city, dst_country, dst_province, dst_city, level='', category=''):
    alarm_id = str(uuid.uuid1())  # uuid
    alarm_type, alarm_2nd_type, alarm_3rd_type = 90000, 90200, 90201
    alarm_grade = 7
    total = 1
    is_external = 0
    status = 0

    if alarm_table.find('dga') != -1:
        alarm_desc = 'dga domain'
        sql_insert_1 = '''REPLACE INTO ''' + alarm_table + ''' (alarm_id, org_id, dga_domain, dga_level, dga_category,
        is_malicious_ip, asset_src_ip, dst_ip, connect_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
        list_1 = (alarm_id, org_id, domain, level, category, '1', src_ip, dst_ip, connect_time)  # 9
    elif alarm_table.find('hostoutreach') != -1:
        alarm_desc = 'black domain'
        sql_insert_1 = '''REPLACE INTO ''' + alarm_table + ''' (alarm_id, org_id, ma_domain, ma_level, ma_category,
        parsing_ip, is_malicious_ip, asset_src_ip, dst_ip, connect_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''
        list_1 = (alarm_id, org_id, domain, level, category, '', '1', src_ip, dst_ip, connect_time)  # 9
    else:
        return

    list_2 = (alarm_id, event_id)  # 2
    list_3 = (alarm_id, org_id, alarm_type, alarm_2nd_type, alarm_3rd_type, alarm_grade, total,
              alarm_desc, connect_time, is_external, src_ip, dst_ip, status,
              src_country, src_province, src_city, dst_country, dst_province, dst_city)  # 19

    try:
        # insert one
        conn = pymysql.connect(host=HOST, port=3306, user=USER, passwd=PASSWORD, db=DB, charset="utf8")
        cur = conn.cursor()

        cur.execute(sql_insert_1, list_1)
        conn.commit()

        sql_insert_2 = '''REPLACE INTO ''' + ALARM_REF_TABLE + ''' (alarm_id, event_id) VALUES (%s, %s)'''
        cur.execute(sql_insert_2, list_2)
        conn.commit()

        sql_insert_3 = '''REPLACE INTO ''' + ALARM_INFO_TABLE + ''' (alarm_id, org_id, alarm_type,
        alarm_2nd_type, alarm_3rd_type, alarm_grade, total, alarm_desc, alarm_time, is_external,
        src_ip, dst_ip, status, src_country, src_province, src_city, dst_country, dst_province, dst_city
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''
        cur.execute(sql_insert_3, list_3)
        conn.commit()
    except Exception as ex_result:
        pass
    finally:
        cur.close()
        conn.close()



if __name__ == "__main__":
    # time_format = "%Y-%m-%d  %H:%M:%S"
    # logging.info('start time: %s !', time.strftime(time_format, time.localtime()))
    options = read_conf(CURRENT_PATH + "/settings_dga.conf")
    ret = load_black_domains()
    black_domains = list(ret[:, 0])
    domain = '0.n1.www1.biz'
    print(run_domain_hostoutreach('0.n1.www1.biz'))
    index = black_domains.index(domain)

    levels = list(ret[:, 1])
    types = list(ret[:, 2])
    level = levels[index]
    type = types[index]
    print(type, level)