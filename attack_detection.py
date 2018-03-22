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


def read_conf():
    cfg = RawConfigParser()
    cfg.read('settings.conf')
    _keys = ['host', 'user', 'passwd', 'port', 'db']
    _options = {_k: cfg.get('mysql', _k).strip() for _k in _keys}
    cal_table_names = cfg.get('mysql', 'cal_table_names').split(',')
    warn_table_names = cfg.get('mysql', 'predict_table_names').split(',')
    hdfs_host = cfg.get('hdfs', 'attack_host').strip()
    solr_host = cfg.get('solr', 'solr_host').strip()
    _options['cal_table_names'] = cal_table_names
    _options['warn_table_names'] = warn_table_names
    _options['attack_host'] = hdfs_host
    _options['solr_host'] = solr_host
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


def get_solr_data(ip_addr, date1, event, cal_table):
    field_names = ['EVENT_ID', 'EVENT_CATEGORY', 'ATTACK_TYPE', 'S_IP', 'SRC_PORT', 'DST_IP', 'DST_PORT',
                   'PROTOCOL', 'EQU_IP', 'EVENT_TIME', 'ACCOUNT', 'ACTION', 'PRIORITY', 'SRC_COUNTRY',
                   'SRC_PROVINCE', 'SRC_CITY', 'DST_COUNTRY', 'DST_PROVINCE', 'DST_CITY', 'ASSETS_NAME',
                   'DST_BUSINESS_SYSTEM', 'DST_ADMIN', 'DST_ADMIN_DEPT', 'EVENT_CONTENT', 'STEP_SIZE', 'BEGIN_TIME',
                   'END_TIME', 'ATTACK_COUNT', 'ALL_BYTE_FLOW', 'ALL_PACKAGE_FLOW', 'AVG_BYTE_FLOW',
                   'AVG_PACKAGE_FLOW', 'MAX_BYTE_FOLW', 'MAX_PACKAGE_FOLW', 'FLOWLINE', 'FLOW', 'FLOW_OF_LINE',
                   'FLOW_OF', 'UP_FLOWLINE', 'UP_FLOW', 'DOWN_FLOWLINE', 'DOWN_FLOW', 'OS_ID', 'EVENT_SOURCE',
                   'EVENT_PROCESS', 'MESSAGE', 'CVE_ID', 'VLUN_NAME', 'VLUN_LEVEL', 'VLUN_TYPE', 'COLLECT_TIME',
                   'CREATE_TIME', 'DESCRIPTION', 'SOLUTION', 'EQU_IP', 'ID', 'ASSETS_TYPE', 'MODEL_ITEM_NAME',
                   'RESULT', 'USER_NAME']
    # [attack_cnt, Predict] + field_names
    _dict = defaultdict(lambda: [0, ''] + [''] * len(field_names))
    solr = Solr('{0}/solr/tda_{1}'.format(ip_addr, date1))
    results = solr.search('EVENT_CATEGORY:{}'.format(event))
    for data in results:
        dst_ip = data['DST_IP']
        _dict[dst_ip][0] += 1

        for _i, name in enumerate(field_names, 2):
            _dict[dst_ip][_i] = data[name]

    values = map(lambda _dip: [date1, _dip] + _dict[_dip], _dict)

    if event == 'ddos_attack':
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


if __name__ == "__main__":
    # options = {'host': '192.168.1.199', 'user': 'root', 'passwd': '1qaz7ujm', 'port': 3306,
    #            'db': 'attack', 'table_names': ['PASSWORD_GUESS_ATTACK_ALARM_INFO_PERDICT'],
    #            'hdfs_host': 'http://192.168.1.192:50070','solr_host': 'http://192.168.1.192:8181'}
    options = read_conf()
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime('%Y%m%d')
    # attack_dir = ['/asap/passwd/', '/asap/web', '/asap/scan', '/asap/program']
    events = ['password_guess', 'web_attack', 'malicious_scan', 'malicious_program', 'ddos_attack', 'log_crack',
              'system_auth', 'error_log', 'vul_used', 'configure_compliance', 'weak_password']
    for _event, _cal_table in zip(events, options['cal_table_names']):
        get_solr_data(options['solr_host'], yesterday, _event, _cal_table)

    # for t in options['alarm_table_names']:
    #     # d = datetime.strptime(get_earliest_date(t, options).split(' ')[0], '%Y-%m-%d')
    #     # while d < today:
    #     get_mysql_data(yesterday, t, options)
