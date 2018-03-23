# coding=utf-8
import sys

sys.path.append('/data/asap/python_package')
import os
import pymysql
from collections import defaultdict
from datetime import datetime, timedelta
from ConfigParser import RawConfigParser
from hdfs.client import Client
from pysolr import Solr
from base_ts import get_Predictions_types_wo_bound
import pandas as pd


def read_conf():
    cfg = RawConfigParser()
    cfg.read('settings.conf')
    _keys = ['host', 'user', 'passwd', 'port', 'db']
    _options = {_k: cfg.get('mysql', _k).strip() for _k in _keys}
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


def get_mysql_data(date1, table1, options):
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DST_ASSET_ID FROM {} WHERE DATE(ALARM_CREATE_TIME)={}".format(table1, date1))
    _dict = defaultdict(lambda: 0)
    for assert_id in cursor.fetchall():
        _dict[assert_id] += 1

    values = map(lambda x: [date1, x, _dict[x]], _dict)

    cursor.executemany(
        'insert into {}(attack_date, assert_id, attack_cnt) values(%s, %s, %s)'.format(table1 + '_PREDICT'), values)
    conn.commit()
    cursor.close()
    conn.close()


def get_hdfs_data(date1, attack_dir, predict_table, options):
    client = Client(options['attack_host'])
    filepath = os.path.join(attack_dir, date1 + '.dat')
    try:
        with client.read(filepath) as fs:
            content = fs.read().strip()
    except:
        return

    priority, attack_type, baseline = [] * 3
    if predict_table in ['passwd', 'scan', 'guess']:
        _dict = defaultdict(lambda: [0, '', '', ''])  # [攻击次数，预警等级，事件类型，业务系统]
        for data_str in content.split('\n'):
            try:
                data_list = data_str.split(',')
                dst_ip = data_list[5]
                _dict[dst_ip][0] += 1  # DST_ASSET_ID/DST_IP
                _dict[dst_ip][1] = data_list[20]  # PRIORITY/WARN_LEVEL
                _dict[dst_ip][2] = data_list[2]  # EVENT_CATEGORY
                _dict[dst_ip][3] = data_list[1]  # DST_BUSINESS_SYSTEM
                # _dict[dst_ip][3] = data_list[3]       # ATTACK_STAGE
            except:
                pass
    elif predict_table in ['web']:
        _dict = defaultdict(lambda: [0, '', '', ''])  # [攻击次数，攻击类型，事件类型，业务系统]
        for data_str in content.split('\n'):
            try:
                data_list = data_str.split(',')
                dst_ip = data_list[5]
                _dict[dst_ip][0] += 1  # DST_ASSET_ID/DST_IP
                _dict[dst_ip][1] = data_list[2]
                _dict[dst_ip][2] = data_list[1]
                _dict[dst_ip][3] = data_list[20]
            except:
                pass
    elif predict_table in ['ddos']:
        _dict = defaultdict(lambda: [0, '', '', ''])  # [攻击次数/峰值流量，攻击类型，事件类型，业务系统]
        for data_str in content.split('\n'):
            try:
                data_list = data_str.split(',')
                dst_ip = data_list[5]
                _dict[dst_ip][0] += data_list[5]  # DST_ASSET_ID/DST_IP   MAX_BYTE_FOLW
                _dict[dst_ip][1] = data_list[2]
                _dict[dst_ip][2] = data_list[1]
                _dict[dst_ip][3] = data_list[20]
                # TODO: 预警基线，开始时间，结束时间，持续时间
            except:
                pass
    elif predict_table in ['log_crack', 'system_auth', 'error_log']:  # 日志破坏，系统提权，错误日志
        # [攻击次数，攻击类型，事件类型，业务系统，目的IP操作系统，EVENT_SOURCE，EVENT_PROCESS，EVENT_NAME/EVENT_ID,事件级别]
        _dict = defaultdict(lambda: [0, '', '', '', '', '', '', '', ''])
        for data_str in content.split('\n'):
            try:
                data_list = data_str.split(',')
                dst_ip = data_list[5]
                _dict[dst_ip][0] += 1  # DST_ASSET_ID/DST_IP
                _dict[dst_ip][1] = data_list[2]
                _dict[dst_ip][2] = data_list[1]
                _dict[dst_ip][3] = data_list[20]
                _dict[dst_ip][4] = data_list[21]
                _dict[dst_ip][5] = data_list[22]
                _dict[dst_ip][6] = data_list[23]
                _dict[dst_ip][7] = data_list[24]
                _dict[dst_ip][8] = data_list[25]
            except:
                pass
    elif predict_table in ['vul']:  # 日志破坏，系统提权，错误日志
        # [攻击次数，攻击类型，事件类型，业务系统，漏洞名称，漏洞类型，漏洞等级，漏洞发现时间，漏洞来源/漏扫设备IP，CVE_ID]
        _dict = defaultdict(lambda: [0, '', '', '', '', '', '', '', ''])
        for data_str in content.split('\n'):
            try:
                data_list = data_str.split(',')
                dst_ip = data_list[5]
                _dict[dst_ip][0] += 1  # DST_ASSET_ID/DST_IP
                _dict[dst_ip][1] = data_list[2]
                _dict[dst_ip][2] = data_list[1]
                _dict[dst_ip][3] = data_list[20]
                _dict[dst_ip][4] = data_list[21]
                _dict[dst_ip][5] = data_list[22]
                _dict[dst_ip][6] = data_list[23]
                _dict[dst_ip][7] = data_list[24]
                _dict[dst_ip][8] = data_list[25]
                _dict[dst_ip][9] = data_list[26]
            except:
                pass
    elif predict_table in ['vul']:  # 日志破坏，系统提权，错误日志
        # [攻击次数，攻击类型，事件类型，业务系统，漏洞名称，漏洞类型，漏洞等级，漏洞发现时间，漏洞来源/漏扫设备IP，CVE_ID]
        _dict = defaultdict(lambda: [0, '', '', '', '', '', '', '', ''])
        for data_str in content.split('\n'):
            try:
                data_list = data_str.split(',')
                dst_ip = data_list[5]
                _dict[dst_ip][0] += 1  # DST_ASSET_ID/DST_IP
                _dict[dst_ip][1] = data_list[2]
                _dict[dst_ip][2] = data_list[1]
                _dict[dst_ip][3] = data_list[20]
                _dict[dst_ip][4] = data_list[21]
                _dict[dst_ip][5] = data_list[22]
                _dict[dst_ip][6] = data_list[23]
                _dict[dst_ip][7] = data_list[24]
                _dict[dst_ip][8] = data_list[25]
                _dict[dst_ip][9] = data_list[26]
            except:
                pass
        values = map(lambda x: [date1, x, _dict[x], priority], _dict)
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cursor = conn.cursor()
    cursor.executemany(
        'insert into {}(attack_date, assert_id, attack_cnt, priority) values(%s, %s, %s, %s)'.format
        (predict_table), values)
    conn.commit()
    cursor.close()
    conn.close()


def get_solr_data(date1, event, cal_table, options):
    field_names = options['field_names']
    # [attack_cnt, Predict] + field_names
    _dict = defaultdict(lambda: [0, ''] + [''] * len(field_names))
    solr = Solr('{0}/solr/{1}_{2}'.format(options['solr_host'], event, date1))
    results = solr.search('EVENT_CATEGORY:{}'.format(event))
    for data in results:
        dst_ip = data['DST_IP']
        _dict[dst_ip][0] += 1

        for _i, name in enumerate(field_names, 2):
            _dict[dst_ip][_i] = data[name]

    values = map(lambda _dip: [date1, _dip] + _dict[_dip], _dict)

    if event == 1006:  # ddos
        values = map(lambda _value: _value[0:2] + [_value[field_names.index('MAX_BYTE_FOLW') + 4]] + _value[3:], values)

    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cursor = conn.cursor()
    cursor.executemany(
        'insert into {0}(attack_date,assert_id,attack_cnt,Predict,{1}) values(%s,%s,%s,%s, {2})'.format(
            cal_table, ','.join(field_names), ','.join(['%s'] * len(field_names))), values)
    conn.commit()
    cursor.close()
    conn.close()


def get_history_data(table_name, options):
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cur = conn.cursor()
    sql = "select * from " + table_name
    cur.execute(sql)
    his_data = cur.fetchall()
    columns = [u'ID', u'Date', u'Type', u'Number', u'Predict1']
    # df_history = pd.DataFrame(list(his_data), columns=columns)
    df_history = pd.DataFrame(list(his_data)).iloc[:, 0:5]
    df_history.columns = columns
    return df_history


def update_data(Predictions, table_name, options):
    conn = pymysql.connect(host=options['host'], user=options['user'], passwd=options['passwd'],
                           port=int(options['port']), db=options['db'], charset='utf8')
    cur = conn.cursor()
    sql = "UPDATE " + table_name + " SET Predict=%s" \
                                   " WHERE attack_date=%s AND assert_id=%s "
    cur.executemany(sql, Predictions)
    conn.commit()
    cur.close()
    conn.close()
    print("finishing update")


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


if __name__ == "__main__":
    # options = {'host': '192.168.1.199', 'user': 'root', 'passwd': '1qaz7ujm', 'port': 3306,
    #            'db': 'SSA_TELECOM',
    #            'hdfs_host': 'http://192.168.1.192:50070','solr_host': 'http://192.168.1.192:8181'}
    options = read_conf()
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime('%Y%m%d')
    # events = ['password_guess', 'web_attack', 'malicious_scan', 'malicious_program', 'ddos_attack', 'log_crack',
    #           'system_auth', 'error_log', 'vul_used', 'configure_compliance', 'weak_password']
    events = [1001, 1002, 1003, 1004, 1006, 2001, 2002, 2003, 3001, 3002, 3003]
    for _event, _cal_table in zip(events, options['cal_table_names']):
        get_solr_data(yesterday, _event, _cal_table, options)

    delete_data(options)

    for cal_table, warn_table in zip(options['cal_table_names'], options['warn_tables_names']):
        df_history_table = get_history_data(cal_table, options)
        Predictions_table = get_Predictions_types_wo_bound(df_history_table, maxar=5, maxma=5, diffn=0, test_size=30,
                                                           multiplier=2)
        update_data(Predictions_table, cal_table, options)
        insert_data(cal_table, warn_table, yesterday, options['field_names'])
