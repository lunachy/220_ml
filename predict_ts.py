# -*- coding: utf-8 -*-
import sys

sys.path.append('/data/asap/python_package')
import os
from datetime import datetime, timedelta
import ConfigParser
import logging
import pymysql
import numpy as np
import pandas as pd
from base_ts import get_Predictions_types_wo_bound

config_parser = ConfigParser.ConfigParser()
config_parser.read('settings.conf')
mysqlHost = config_parser.get("mysql", "host")
user = config_parser.get("mysql", "user")
password = config_parser.get("mysql", "passwd")
db_text = config_parser.get("mysql", "db")
cal_tables = config_parser.get("mysql", "cal_table_names").split(',')
warn_tables = config_parser.get("mysql", "warn_table_names").split(',')
field_names = config_parser.get("common", "field_names").split(',')


# logFile = config_parser.get("logging", "logging_dir_path")


# def init_logging(logFilename):
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
#         datefmt='%Y-%m-%d %A %H:%M:%S',
#         filename=logFilename,
#         filemode='a')
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)
#     solrLogger = logging.getLogger('pysolr')
#     solrLogger.setLevel(logging.ERROR)
#     connectionLogger = logging.getLogger('requests')
#     connectionLogger.setLevel(logging.ERROR)


def get_history_data(table_name):
    conn = pymysql.connect(host=mysqlHost, port=3306, user=user, passwd=password, db=db_text, charset="utf8")
    cur = conn.cursor()
    sql = "select * from " + table_name
    cur.execute(sql)
    his_data = cur.fetchall()
    columns = [u'ID', u'Date', u'Type', u'Number', u'Predict1']
    # df_history = pd.DataFrame(list(his_data), columns=columns)
    df_history = pd.DataFrame(list(his_data)).iloc[:, 0:5]
    df_history.columns = columns
    return df_history


def update_data(Predictions, table_name):
    conn = pymysql.connect(host=mysqlHost, port=3306, user=user, passwd=password, db=db_text, charset="utf8")
    cur = conn.cursor()
    sql = "UPDATE " + table_name + " SET Predict=%s" \
                                   " WHERE attack_date=%s AND assert_id=%s "
    cur.executemany(sql, Predictions)
    conn.commit()
    cur.close()
    conn.close()
    print("finishing update")


def delete_data(warn_table):
    base_table = 'early_warn_info'
    conn = pymysql.connect(host=mysqlHost, port=3306, user=user, passwd=password, db=db_text, charset="utf8")
    cur = conn.cursor()
    cur.execute('select ID from {}'.format(warn_table))
    results = cur.fetchall()
    for _id in results:
        cur.execute('delete from {} where ID={}'.format(base_table, _id[0]))
    cur.execute('delete from {}'.format(warn_table))
    conn.commit()
    cur.close()
    conn.close()


def insert_data(cal_table, warn_table, date1, field_names):
    # date1: 20180203
    date_dt = datetime.strptime(date1, '%Y%m%d')
    base_table = 'early_warn_info'
    base_sql = 'insert into {}({}) values({})'
    base_names = 'WARN_NAME,CREATE_TIME,DEVICE,WARN_LEVEL,WARN_TYPE,GROUP_ID'

    # complete extra fields
    field_names = ['uuid', 'attack_date', 'assert_id', 'attack_cnt', 'Predict'] + field_names
    conn = pymysql.connect(host=mysqlHost, port=3306, user=user, passwd=password, db=db_text, charset="utf8")
    cur = conn.cursor()
    cur.execute('select * from {} where DATE(attack_date)={}'.format(cal_table, date1))
    results = cur.fetchall()
    if warn_table == 'network_in_early_warn_info':
        _main_names = 'ID,ATTACK_TIME,ATTACK_TYPE,EVENT_TYPE'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'EVENT_CATEGORY']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    value.insert(0, ret_id)
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
    elif warn_table == 'web_attack_early_warn_info':
        _main_names = 'ID,ATTACK_TYPE,ATTACK_TIME,EVENT_TYPE'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['ATTACK_TYPE', 'EVENT_CATEGORY']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    value.insert(0, ret_id)
                    value.insert(2, pred_time)
                    cur.execute(sql, value)
    elif warn_table == 'ddos_at_early_warn_info':
        _main_names = 'ID,ATTACK_TYPE,PEAK_FLOW,EXCEED_PERCENT,EVENT_TYPE'  # START_TIME,END_TIME,CONTINUE_TIME
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['ATTACK_TYPE', 'FLOWLINE', 'EVENT_CATEGORY']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    value.insert(0, ret_id)
                    value.insert(2, _p)
                    value[3] = round((_p * 1.0 / value[3] - 1) * 100, 2)  # 计算超过基线值，取两位小数
                    cur.execute(sql, value)
    elif warn_table == 'system_attack_early_warn_info':
        _main_names = 'ID,LOG_TYPE,OS_TYPE,EVENT_SOURCE,EVENT_PROCESS,EVENT_NAME,TASK_TYPE'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'OS_ID', 'EVENT_SOURCE', 'EVENT_PROCESS', 'EVENT_ID', 'PRIORITY']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    value.insert(0, ret_id)
                    cur.execute(sql, value)
    elif warn_table == 'vulnerability_early_warn_info':
        _main_names = 'ID,FIND_TIME,VLUN_NAME,VLUN_TYPE,VLUN_LEVEL,EXTERNAL_VUL_ID'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['CREATE_TIME', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['CREATE_TIME', 'VLUN_NAME', 'VLUN_TYPE', 'VLUN_LEVEL', 'CVE_ID']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    value.insert(0, ret_id)
                    cur.execute(sql, value)
    elif warn_table == 'compliance_early_warn_info':
        _main_names = 'ID,EVENT_TYPE'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['CREATE_TIME', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_CATEGORY']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    # pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    # value.insert(1, pred_time)
                    value.insert(0, ret_id)
                    cur.execute(sql, value)
    elif warn_table == 'weak_password_early_warn_info':
        _main_names = 'ID,EVENT_TYPE,GROUP_ID,USER_NAME'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['CREATE_TIME', 'assert_id', 'PRIORITY', 'EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_CATEGORY', 'DST_BUSINESS_SYSTEM', 'USER_NAME']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    # pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    # value.insert(1, pred_time)
                    value.insert(0, ret_id)
                    cur.execute(sql, value)
    conn.commit()
    cur.close()
    conn.close()


if __name__ == '__main__':
    # init_logging(os.path.join(logFile, "predict" + ".log"))
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime('%Y%m%d')
    # [('password_guess_cal_info', 'network_in_early_warn_info'),
    #  ('web_attack_cal_info', 'web_attack_early_warn_info'),
    #  ('malicious_scan_cal_info', 'network_in_early_warn_info'),
    #  ('malicious_program_cal_info', 'network_in_early_warn_info'),
    #  ('ddos_attack_cal_info', 'ddos_at_early_warn_info'),
    #  ('log_crack_cal_info', 'network_in_early_warn_info'),
    #  ('system_auth_cal_info', 'network_in_early_warn_info'),
    #  ('error_log_cal_info', 'network_in_early_warn_info'),
    #  ('vul_used_cal_info', 'vulnerability_early_warn_info'),
    #  ('configure_compliance_cal_info', 'compliance_early_warn_info'),
    #  ('weak_password_cal_info', 'weak_password_early_warn_info')]

    for warn_table in warn_tables:
        delete_data(warn_table)

    for cal_table, warn_table in zip(cal_tables, warn_tables):
        # logging.info("process %s data" % cal_table)
        df_history_table = get_history_data(cal_table)
        # logging.info('the shape of %s.', df_history_table.shape)
        Predictions_table = get_Predictions_types_wo_bound(df_history_table, maxar=5, maxma=5, diffn=0, test_size=30,
                                                           multiplier=2)
        # logging.info(Predictions_table)
        update_data(Predictions_table, cal_table)
        # logging.info("finished processing %s data" % cal_table)
        insert_data(cal_table, warn_table, yesterday, field_names)
    # logging.info("finished processing")
