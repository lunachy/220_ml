# coding=utf-8
import sys

sys.path.append('/hadoop2/asap/ssa/python_package')
import os
import pymysql
from collections import defaultdict
from datetime import datetime, timedelta
from ConfigParser import RawConfigParser
# from hdfs.client import Client
from pysolr import Solr
from random import uniform


# from base_ts import get_Predictions_types_wo_bound
# import pandas as pd


def read_conf():
    cfg = RawConfigParser()
    cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    _keys = ['host', 'user', 'passwd', 'port', 'db']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    # _options = {_k: cfg.get('mysql', _k).strip() for _k in _keys} # >= python 2.7
    cal_table_names = cfg.get('mysql', 'cal_table_names').split(',')
    warn_table_names = cfg.get('mysql', 'warn_table_names').split(',')
    hdfs_host = cfg.get('hdfs', 'attack_host').strip()
    solr_host = cfg.get('solr', 'solr_host').strip()
    field_names = cfg.get('common', 'field_names').split(',')
    _options['cal_table_names'] = cal_table_names
    _options['warn_table_names'] = warn_table_names
    _options['hdfs_host'] = hdfs_host
    _options['solr_host'] = solr_host
    _options['field_names'] = field_names
    return _options


def get_earliest_date(table1, options):
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cursor = conn.cursor()
    cursor.execute("SELECT min(ALARM_CREATE_TIME) from {}".format(table1))
    _datetime = cursor.fetchone()

    cursor.close()
    conn.close()
    return _datetime


def get_solr_data(date1, event, cal_table, options):
    field_names = options['field_names']
    # [attack_cnt, Predict] + field_names
    _dict = defaultdict(lambda: [0, ''] + [''] * len(field_names))
    solr = Solr('{0}/solr/{1}_{2}'.format(options['solr_host'], event, date1))
    results = solr.search('*:*', rows=10000)
    print event, len(results)
    for data in results:
        if 'DST_IP' in data:
            dst_ip = data['DST_IP']
            _dict[dst_ip][0] += 1
            for _i, name in enumerate(field_names, 2):
                if name in data:
                    _dict[dst_ip][_i] = data[name]
    values = map(lambda _dip: [date1, _dip] + _dict[_dip], _dict)
    if event == 'ddos_attack':
        values = map(lambda _value: _value[0:2] + [_value[field_names.index('MAX_BYTE_FOLW') + 4]] + _value[3:], values)
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cursor = conn.cursor()
    cursor.execute('delete from {} where DATE(attack_date)={}'.format(cal_table, date1))
    conn.commit()
    cursor.executemany(
        'insert into {0}(attack_date,assert_id,attack_cnt,Predict,{1}) values(%s,%s,%s,%s, {2})'.format(
            cal_table, ','.join(field_names), ','.join(['%s'] * len(field_names))), values)
    conn.commit()
    cursor.close()
    conn.close()


# def get_history_data(table_name, options):
#     conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
#                            port=int(options['port']), db=options['db'], charset='utf8')
#     cur = conn.cursor()
#     sql = "select * from " + table_name
#     cur.execute(sql)
#     his_data = cur.fetchall()
#     columns = [u'ID', u'Date', u'Type', u'Number', u'Predict1']
#     # df_history = pd.DataFrame(list(his_data ), columns=columns)
#     df_history = pd.DataFrame(list(his_data)).iloc[:, 0:5]
#     df_history.columns = columns
#     cur.close()
#     conn.close()
#     return df_history


def update_predict_data(table_name, options, date1):
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cur = conn.cursor()
    select_sql = "select assert_id,attack_cnt,Predict from {} where DATE(attack_date)={}"
    update_sql = "UPDATE {} SET Predict='{}' WHERE DATE(attack_date)={} AND assert_id='{}'"
    cur.execute(select_sql.format(table_name, date1))
    date1_data = cur.fetchall()

    date1_p = datetime.strptime(date1, '%Y%m%d')
    date2 = (date1_p - timedelta(days=1)).strftime('%Y%m%d')
    cur.execute(select_sql.format(table_name, date2))
    date2_data = cur.fetchall()

    for d1 in date1_data:
        for d2 in date2_data:
            if d1[0] == d2[0]:
                pred_value = [d1[1]] + map(int, d2[2].strip('[').strip(']').split(','))
                cur.execute(update_sql.format(table_name, str(pred_value[:30]), date1, d1[0]))
                conn.commit()
    for d1 in date1_data:
        if not d1[2]:
            pred_v = int(uniform(0.9,1.1)*d1[1])
            cur.execute(update_sql.format(table_name, str([pred_v]), date1, d1[0]))
            conn.commit()

    cur.close()
    conn.close()


# def update_data(Predictions, table_name, options):
#     conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
#                            port=int(options['port']), db=options['db'], charset='utf8')
#     cur = conn.cursor()
#     sql = "UPDATE " + table_name + " SET Predict=%s WHERE attack_date=%s AND assert_id=%s "
#     cur.executemany(sql, Predictions)
#     conn.commit()
#     cur.close()
#     conn.close()
#     print("finishing update")


def delete_data(options):
    base_table = 'early_warn_info'
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cur = conn.cursor()
    for warn_table in options['warn_table_names']:
        cur.execute('select ID from {}'.format(warn_table))
        results = cur.fetchall()
        for _id in results:
            cur.execute('delete from {} where ID={}'.format(base_table, _id[0]))
        cur.execute('delete from {}'.format(warn_table))
    conn.commit()
    cur.close()
    conn.close()


def insert_data(cal_table, warn_table, date1, options):
    # date1: 20180203
    date_dt = datetime.strptime(date1, '%Y%m%d')
    field_names = options['field_names']
    base_table = 'early_warn_info'
    base_sql = 'insert into {}({}) values({})'
    base_names = 'WARN_NAME,CREATE_TIME,DEVICE,WARN_LEVEL,WARN_TYPE,GROUP_ID'

    # complete extra fields
    field_names = ['uuid', 'attack_date', 'assert_id', 'attack_cnt', 'Predict'] + field_names
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cur = conn.cursor()
    cur.execute('select * from {} where DATE(attack_date)={}'.format(cal_table, date1))
    results = cur.fetchall()
    if not results:
        return
    if warn_table == 'network_in_early_warn_info':
        _main_names = 'ID,ATTACK_TIME,ATTACK_TYPE,EVENT_TYPE'
        for r in results:
            pred = map(int, r[4].strip('[').strip(']').split(','))
            for _i, _p in enumerate(pred, 1):
                if _p:
                    # insert into early_warn_info
                    sql = base_sql.format(base_table, base_names, ','.join(['%s'] * len(base_names.split(','))))
                    _fields = ['EVENT_TYPE', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_TYPE', 'EVENT_TYPE']
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
                    _fields = ['EVENT_TYPE', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['ATTACK_TYPE', 'EVENT_TYPE']
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
                    _fields = ['EVENT_TYPE', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['ATTACK_TYPE', 'FLOWLINE', 'EVENT_TYPE']
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
                    _fields = ['EVENT_TYPE', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    print value
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_TYPE', 'OS_ID', 'EVENT_SOURCE', 'EVENT_PROCESS', 'EVENT_ID', 'SERVERITY']
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
                    _fields = ['CREATE_TIME', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
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
                    _fields = ['CREATE_TIME', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_TYPE']
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
                    _fields = ['CREATE_TIME', 'assert_id', 'SERVERITY', 'EVENT_TYPE', 'DST_BUSINESS_SYSTEM']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    value.insert(1, pred_time)
                    cur.execute(sql, value)
                    ret_id = cur.lastrowid

                    # insert into main warn info
                    sql = base_sql.format(warn_table, _main_names, ','.join(['%s'] * len(_main_names.split(','))))
                    _fields = ['EVENT_TYPE', 'DST_BUSINESS_SYSTEM', 'USER_NAME']
                    value = map(lambda x: r[field_names.index(x)], _fields)
                    # pred_time = datetime.strftime(date_dt + timedelta(_i), '%Y%m%d')
                    # value.insert(1, pred_time)
                    value.insert(0, ret_id)
                    cur.execute(sql, value)
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    # options = {'host': '192.168.1.199', 'user': 'root', 'passwd': '1qaz7ujm', 'port': 3306,
    #            'db': 'SSA_TELECOM',
    #            'hdfs_host': 'http://192.168.1.192:50070','solr_host': 'http://192.168.1.192:8181'}
    options = read_conf()
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime('%Y%m%d')
    # yesterday = today.strftime('%Y%m%d')
    # events = ['1001', '1002', '1003', '1004', '1006', '2001', '2002', '2003', '3001', '3002', '3003']
    events = ['password_guessing_attack', 'web_attack', 'malicious_scan_attack', 'malicious_program_attack',
              'ddos_attack', 'log_damage_detection', 'system_privilege_detection',
              'error_log_detection', 'vuln_used', 'conf_compliance_used', 'weak_pwd_used']
    for _event, _cal_table in zip(events, options['cal_table_names']):
        try:
            get_solr_data(yesterday, _event, _cal_table, options)
        except Exception, e:
            print _event, e

    delete_data(options)

    for cal_table, warn_table in zip(options['cal_table_names'], options['warn_table_names']):
        # df_history_table = get_history_data(cal_table, options)
        # Predictions_table = get_Predictions_types_wo_bound(df_history_table, maxar=5, maxma=5, diffn=0, test_size=30,
        #                                                    multiplier=2)
        # update_data(Predictions_table, cal_table, options)
        update_predict_data(cal_table, options, yesterday)
        insert_data(cal_table, warn_table, yesterday, options)
