# -*- coding: utf-8 -*-
import sys
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURRENT_PATH, 'packages_centos7'))
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
from utils_all import init_logging

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


def prepare_token(token):
    # pattern = r'''^(25[0-5]|2[0-4]\d|[0-1]?\d?\d)(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}$'''
    token, _ = re.subn(r'\d+', "8", token)
    my_extract = tldextract.TLDExtract(suffix_list_urls=None)
    val = my_extract(token)
    token_1 = ''.join(val.subdomain.split('.'))
    token_2 = val.domain + val.suffix
    if DOMAIN_FLAG:
        tokens = [token_1, token_2]
    else:
        tokens = [token_1]
    return tokens


def entropy(token):
    """compute entropy"""
    pair, lns = Counter(token), float(len(token))
    return -sum(count / lns * math.log(count / lns, 2) for count in pair.values())


def get_dga_ml_features(token, alexa_vocab, alexa_counts, word_vocab, word_counts):
    """extract features"""
    alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0, vocabulary=alexa_vocab)
    word_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0, vocabulary=word_vocab)
    x_length = len(token)
    x_entropy = entropy(token)
    x_alexa_grams = alexa_counts * alexa_vc.transform([token]).T
    x_word_grams = word_counts * word_vc.transform([token]).T
    x_diff = x_alexa_grams.item() - x_word_grams.item()
    features = np.array([x_length, x_entropy, x_alexa_grams.item(), x_word_grams.item(), x_diff]).reshape(1, -1)
    return features


def load_models(root_dir):
    """load pretrained models"""
    clf_et = joblib.load(os.path.join(root_dir, 'models', 'domain_dga_ExtraTree.pkl'))
    alexa_vocab = joblib.load(os.path.join(root_dir, 'counts', 'domain_dga_vocab_alexa.txt'))
    alexa_counts = np.load(os.path.join(root_dir, 'counts', 'domain_dga_alexa_counts.npz'))['alexa_counts']
    word_vocab = joblib.load(os.path.join(root_dir, 'counts', 'domain_dga_vocab_word.txt'))
    word_counts = np.load(os.path.join(root_dir, 'counts', 'domain_dga_word_counts.npz'))['dict_counts']
    return clf_et, alexa_vocab, alexa_counts, word_vocab, word_counts


def run_predict_ml(domain, alexa_vocab, alexa_counts, word_vocab, word_counts, clf_et):
    """get predictions"""
    tokens = prepare_token(domain)
    predictions = []
    for token in tokens:
        x_ml_features = get_dga_ml_features(token, alexa_vocab, alexa_counts, word_vocab, word_counts)
        y_pred_et = clf_et.predict(x_ml_features)[0]
        pred_et = 0 if y_pred_et == 'legit' else 1
        predictions.append(pred_et)
    prediction = 1 if sum(predictions) == len(predictions) else 0
    # prediction = 1 if sum(predictions) > 0 else 0
    return prediction


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

                        # domains = set()
                        # if 'dns' in data:
                        #     dns_data = data['dns']
                        #     if 'query' in dns_data:
                        #         if 'rrname' in dns_data['query']:
                        #             domains.add(str(dns_data['query']['rrname']))
                        # if 'answer' in dns_data:
                        #     if 'rrname' in dns_data['answer']:
                        #         domains.add(str(dns_data['answer']['rrname']))
                        #     if 'answers' in dns_data['answer']:
                        #         answers = dns_data['answer']['answers']
                        #         for ans in answers:
                        #             if 'rrname' in ans:
                        #                 domains.add(ans['rrname'])
                        # domains = list(domains)
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
        logging.exception(str(ex_result))
    finally:
        cur.close()
        conn.close()


def run_from_kafka():
    """process run"""
    clf_gb, alexa_vocab, alexa_counts, word_vocab, word_counts = load_models(CURRENT_PATH)

    # auto_offset_reset should set to 'latest' in real situation, 'earliest' just for test
    try:
        kafka_flag = 'latest' if KAFKA_FLAG else 'earliest'
        consumer = KafkaConsumer(TOPIC, bootstrap_servers=KAFKA_ADDR, auto_offset_reset=kafka_flag)
    except:
        logging.error('connect kafka failed, msg: %s', traceback.format_exc())
        sys.exit()

    while 1:
        for count, msg in enumerate(consumer, 1):
            org_id, event_id, domain, src_ip, dst_ip, connect_time, src_country, src_province, src_city, \
            dst_country, dst_province, dst_city = load_data_super(msg.value)

            if domain:
                result = run_predict_ml(domain, alexa_vocab, alexa_counts, word_vocab, word_counts, clf_gb)
                if result == 1:
                    logging.info('flag of domain %s is %s ' % (domain, result))
                    insert_mysql(ALARM_DGA_TABLE, org_id, event_id, domain, src_ip, dst_ip, connect_time,
                                 src_country, src_province, src_city, dst_country, dst_province, dst_city)

                result1 = run_domain_hostoutreach(domain)
                if result1 == 1:
                    index = black_domains.index(domain)
                    level = levels[index]
                    type = types[index]
                    insert_mysql(ALARM_HOSTOUTREACH, org_id, event_id, domain, src_ip, dst_ip, connect_time,
                                 src_country, src_province, src_city, dst_country, dst_province, dst_city, level, type)


if __name__ == "__main__":
    init_logging(LOGFILENAME)
    # time_format = "%Y-%m-%d  %H:%M:%S"
    # logging.info('start time: %s !', time.strftime(time_format, time.localtime()))
    options = read_conf(CURRENT_PATH + "/settings_dga.conf")
    ret = load_black_domains()
    black_domains = list(ret[:, 0])
    levels = list(ret[:, 1])
    types = list(ret[:, 2])
    run_from_kafka()
