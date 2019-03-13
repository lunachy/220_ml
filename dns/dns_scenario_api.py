# -*- coding: utf-8 -*-
import sys
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURRENT_PATH, 'packages_centos7'))
import json
import traceback
import uuid
import logging
import ConfigParser
from kafka import KafkaConsumer
import pymysql
import logging.handlers
import requests

CONFIG_PARSER = ConfigParser.ConfigParser()
CONFIG_PARSER.read(CURRENT_PATH + "/settings_api.conf")
HOST = CONFIG_PARSER.get("mysql", "host")
PORT = CONFIG_PARSER.get("mysql", "port")
USER = CONFIG_PARSER.get("mysql", "user")
PASSWORD = CONFIG_PARSER.get("mysql", "password")
DB = CONFIG_PARSER.get("mysql", "db")
LOGFILENAME = CURRENT_PATH + '/dns_scenario.log'
KAFKA_ADDR = CONFIG_PARSER.get("kafka", "kafka_addr")
TOPIC = CONFIG_PARSER.get("kafka", "topic")
ALARM_HOSTOUTREACH = CONFIG_PARSER.get("mysql", "alarm_hostoutreach")
ALARM_DGA_TABLE = CONFIG_PARSER.get("mysql", "alarm_dga")
ALARM_REF_TABLE = CONFIG_PARSER.get("mysql", "alarm_ref_event")
ALARM_INFO_TABLE = CONFIG_PARSER.get("mysql", "alarm_info")
TOKEN = CONFIG_PARSER.get("api", "token")
API_IP = CONFIG_PARSER.get("api", "ip")
API_ADDR = CONFIG_PARSER.get("api", "addr")

log = logging.getLogger(__file__[:-3])


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


# import ssl
# import Properties
# from httplib import HTTPSConnection
# import base64
#
# def send(path):
#     port = Properties.SSLPORT
#     host = Properties.HOST
#     url = host + ":" + str(port) + path
#     context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
#     context.load_cert_chain(Properties.PRIVATE_KEY, **{"password":Properties.PASSWORD})
#
#     data = '123'
#     httpsConn = HTTPSConnection(host, port, None, None, "", 60000, "", context)
#     httpsConn.request("POST", path, data, {"Authorization": "Basic " + base64.encodestring("12345678")})
#     res = httpsConn.getresponse()
#     print res.status,res.reason, res.getheaders(), res.read()

def get_domaininfo_from_api(domain):
    HEADERS = {"NS-NTIP-KEY": TOKEN}
    REST_URL = "https://{}:{}/api/v4/objects/domain-details/".format(API_IP, API_ADDR)

    r = requests.get(REST_URL, headers=HEADERS, params={'query': domain, 'type': 'indicator'})
    result = r.json()
    if result:
        level = result['threat_level']
        type = result['threat_types'][0]
        return level, type
    else:
        return None, None


def load_data(item):
    """load json data"""
    flag = False
    org_id = ''
    dga_domain = ''
    asset_src_ip = ''
    dst_ip = ''
    event_id = ''
    connect_time = ''
    try:
        data = json.loads(item.split("///")[1])
        if 'alert' in data:
            if 'signature_id' in data['alert']:
                if data['alert']['signature_id'] == 6000001:
                    flag = True

        if flag:
            if 'flow_id ' in data:
                event_id = data['flow_id ']
            if 'timestamp' in data:
                connect_time = data['timestamp']
            if 'src_ip' in data:
                asset_src_ip = data['src_ip']
            if 'dest_ip' in data:
                dst_ip = data['dest_ip']
            if 'dns' in data:
                dns_data = data['dns']
                if 'query' in dns_data:
                    if 'rrname' in dns_data['query'][0]:
                        dga_domain = str(dns_data['query'][0]['rrname'])
    except Exception as ex_results:
        # logging.exception(str(ex_results))
        pass
    return org_id, event_id, dga_domain, asset_src_ip, dst_ip, connect_time


def load_data_super(item):
    """load json data"""
    flag = False
    org_id = ''
    dga_domain = ''
    asset_src_ip = ''
    dst_ip = ''
    event_id = ''
    connect_time = ''
    src_country = ''
    src_province = ''
    src_city = ''
    dst_country = ''
    dst_province = ''
    dst_city = ''
    try:
        data = json.loads(item)
        if 'eventId_s' in data:
            if data['eventId_s'] == "6000001":
                flag = True
        if flag:
            if 'uuid_s' in data:
                event_id = data['uuid_s']
            if 'dstOrgId_s' in data:
                org_id = data['dstOrgId_s']
            if 'logTime_dt' in data:
                connect_time = data['logTime_dt']
            if 'srcIp_s' in data:
                asset_src_ip = data['srcIp_s']
            if 'dstIp_s' in data:
                dst_ip = data['dstIp_s']
            if 'domain_s' in data:
                dga_domain = str(data['domain_s'])
            if 'srcCountry_s' in data:
                src_country = data['srcCountry_s']
            if 'srcProvince_s' in data:
                src_province = data['srcProvince_s']
            if 'srcCity_s' in data:
                src_city = data['srcCity_s']
            if 'dstCountry_s' in data:
                dst_country = data['dstCountry_s']
            if 'dstProvince_s' in data:
                dst_province = data['dstProvince_s']
            if 'dstCity_s' in data:
                dst_city = data['dstCity_s']
    except Exception as ex_results:
        # logging.exception(str(ex_results))
        pass
    return org_id, event_id, dga_domain, asset_src_ip, dst_ip, connect_time, src_country, src_province, src_city, \
           dst_country, dst_province, dst_city


def insert_mysql(alarm_table, org_id, event_id, domain, src_ip, dst_ip, connect_time, src_country,
                 src_province, src_city, dst_country, dst_province, dst_city, level='', category=''):
    alarm_id = str(uuid.uuid1())  # uuid
    alarm_type, alarm_2nd_type, alarm_3rd_type = 90000, 90200, 90201
    alarm_grade = 7
    total = 1
    is_external = 0
    status = 0

    if not level: level = ''
    if not category: category = ''

    if alarm_table.find('hostoutreach') != -1:
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
        logging.exception(str(ex_result))
    finally:
        cur.close()
        conn.close()


def run_from_kafka():
    # auto_offset_reset should set to 'latest' in real situation, 'earliest' just for test
    try:
        kafka_flag = 'latest'
        consumer = KafkaConsumer(TOPIC, bootstrap_servers=KAFKA_ADDR, auto_offset_reset=kafka_flag)
    except:
        logging.error('connect kafka failed, msg: %s', traceback.format_exc())
        sys.exit()

    while 1:
        for count, msg in enumerate(consumer, 1):
            org_id, event_id, domain, src_ip, dst_ip, connect_time, src_country, src_province, src_city, \
            dst_country, dst_province, dst_city = load_data_super(msg.value)

            if domain:
                level, type = get_domaininfo_from_api(domain)
                if level:
                    insert_mysql(ALARM_HOSTOUTREACH, org_id, event_id, domain, src_ip, dst_ip, connect_time,
                                 src_country, src_province, src_city, dst_country, dst_province, dst_city, level, type)


if __name__ == "__main__":
    init_logging(LOGFILENAME)
    run_from_kafka()
