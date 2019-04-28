#!/usr/bin/python
# coding=utf-8

from selenium import webdriver
from time import sleep
import re
import sys
import os
import pymysql
from pymysql import IntegrityError
from configparser import RawConfigParser
import pysnooper
import traceback


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


OPTIONS = read_conf('settings.conf')


def handle_mysql(tb=''):
    conn = pymysql.connect(**OPTIONS)
    cur = conn.cursor()
    sql = 'select * from {}'.format(tb)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def scroll_end(driver, lang='cn'):
    if lang == 'en':
        return
    height = driver.execute_script('return document.body.scrollHeight')
    step = 500
    scrolled_height = 0
    while scrolled_height < height:
        scrolled_height += step
        driver.execute_script("window.scrollTo(0,{})".format(scrolled_height))
        sleep(2)


def get_chrome_option(lang):
    if lang == 'cn':
        chrome_options = webdriver.ChromeOptions()
        prefs = {
            "translate_whitelists": {"en": "zh-CN"},
            "translate": {"enabled": "true"}
        }
        chrome_options.add_experimental_option("prefs", prefs)
    else:
        chrome_options = None
    return chrome_options


def get_tactics(lang, options=OPTIONS):
    entry_url = 'https://attack.mitre.org/tactics/enterprise/'
    driver = webdriver.Chrome(options=get_chrome_option(lang))
    driver.get(entry_url)
    sleep(2)
    scroll_end(driver, lang)
    element = driver.find_element_by_xpath("//tbody[@class='bg-white']")
    attack_tids = []
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    tb = 'mitre_tactics'
    for tr in element.find_elements_by_tag_name('tr'):
        value = list(map(lambda td: td.text, tr.find_elements_by_tag_name('td')))
        attack_tids.append(value[0])
        try:
            if lang == 'cn':
                sql_str = 'update {} set name_cn="{}",desc_cn="{}" where tid="{}"'.format(tb, value[1], value[2],
                                                                                          value[0])
                cur.execute(sql_str)
            else:
                fields = 'tid, name_en, desc_en'
                sql_str = 'insert into {}({}) values({})'.format(tb, fields, ','.join(['%s'] * len(fields.split(','))))
                cur.execute(sql_str, value)
            conn.commit()
        except:
            print(traceback.format_exc())
    cur.close()
    conn.close()
    driver.quit()
    return attack_tids


def get_tech_ids():
    tech_url = 'https://attack.mitre.org/techniques/'
    tech_ids = []
    driver = webdriver.Chrome()
    driver.get(tech_url)
    element = driver.find_element_by_xpath("//tbody[@class='bg-white']")
    for tr in element.find_elements_by_tag_name('tr'):
        tech_id, name, desc = list(map(lambda td: td.text, tr.find_elements_by_tag_name('td')))
        tech_ids.append(tech_id)
    print(tech_ids, len(tech_ids))
    return tech_ids


class DB(object):
    def __init__(self, options=OPTIONS):
        self.conn = pymysql.connect(**options)
        self.cur = self.conn.cursor()

    def __enter__(self):
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()
        self.cur.close()


def select_mysql(tb, sel_fileds, wh_fileds, wh_values):
    # fields = 'tid, name_en, attack_id'
    wh_fileds = wh_fileds.split(',')
    wh_str = ' and '.join([str(f) + '="' + str(v) + '"' for f, v in zip(wh_fileds, wh_values)])
    if wh_str:
        wh_str = 'where ' + wh_str
    sql = 'select {} from {} {}'.format(sel_fileds, tb, wh_str)
    with DB() as cur:
        cur.execute(sql)
        result = cur.fetchall()
    return result


def insert_mysql(tb, fields, values):
    # fields = 'tid,name_en,attack_id'
    sql = 'insert into {}({}) values({})'.format(tb, fields, ','.join(['%s'] * len(values)))
    with DB() as cur:
        try:
            cur.execute(sql, values)
        except Exception as e:
            if e.args[0] == 1062:  # Duplicate entry
                pass
            else:
                print(traceback.format_exc())


def update_mysql(tb, set_fields, set_values, wh_fileds, wh_values):
    # fields = 'tid,name_en,attack_id'
    set_fields = set_fields.split(',')
    wh_fileds = wh_fileds.split(',')
    set_str = ','.join([str(f) + '="' + str(v) + '"' for f, v in zip(set_fields, set_values)])
    wh_str = ' and '.join([str(f) + '="' + str(v) + '"' for f, v in zip(wh_fileds, wh_values)])
    sql = 'update {} set {} where {}'.format(tb, set_str, wh_str)
    with DB() as cur:
        cur.execute(sql)


def get_techniques(tech_ids, lang, options=OPTIONS):
    driver = webdriver.Chrome(options=get_chrome_option(lang))
    tech_ids = ['T1156', 'T1134', 'T1015', 'T1087', 'T1098', 'T1182', 'T1103', 'T1155', 'T1017', 'T1138', 'T1010',
                'T1123', 'T1131', 'T1119', 'T1020', 'T1139', 'T1009', 'T1197', 'T1067', 'T1217', 'T1176', 'T1110',
                'T1088', 'T1042', 'T1146', 'T1115', 'T1191', 'T1116', 'T1059', 'T1043', 'T1092', 'T1223', 'T1109',
                'T1122', 'T1090', 'T1196', 'T1136', 'T1003', 'T1081', 'T1214', 'T1094', 'T1024', 'T1002', 'T1132',
                'T1022', 'T1213', 'T1005', 'T1039', 'T1025', 'T1001', 'T1074', 'T1030', 'T1207', 'T1140', 'T1089',
                'T1175', 'T1038', 'T1073', 'T1172', 'T1189', 'T1157', 'T1173', 'T1114', 'T1106', 'T1129', 'T1048',
                'T1041', 'T1011', 'T1052', 'T1190', 'T1203', 'T1212', 'T1211', 'T1068', 'T1210', 'T1133', 'T1181',
                'T1008', 'T1083', 'T1107', 'T1222', 'T1006', 'T1044', 'T1187', 'T1144', 'T1061', 'T1200', 'T1158',
                'T1147', 'T1143', 'T1148', 'T1179', 'T1062', 'T1183', 'T1054', 'T1066', 'T1070', 'T1202', 'T1056',
                'T1141', 'T1130', 'T1118', 'T1208', 'T1215', 'T1142', 'T1159', 'T1160', 'T1152', 'T1161', 'T1149',
                'T1171', 'T1168', 'T1162', 'T1037', 'T1177', 'T1185', 'T1036', 'T1031', 'T1112', 'T1170', 'T1188',
                'T1104', 'T1026', 'T1079', 'T1128', 'T1046', 'T1126', 'T1135', 'T1040', 'T1050', 'T1096', 'T1027',
                'T1137', 'T1075', 'T1097', 'T1174', 'T1201', 'T1034', 'T1120', 'T1069', 'T1150', 'T1205', 'T1013',
                'T1086', 'T1145', 'T1057', 'T1186', 'T1093', 'T1055', 'T1012', 'T1163', 'T1164', 'T1108', 'T1060',
                'T1121', 'T1117', 'T1219', 'T1076', 'T1105', 'T1021', 'T1018', 'T1091', 'T1014', 'T1085', 'T1053',
                'T1029', 'T1113', 'T1180', 'T1064', 'T1063', 'T1101', 'T1167', 'T1035', 'T1058', 'T1166', 'T1051',
                'T1023', 'T1178', 'T1218', 'T1216', 'T1198', 'T1045', 'T1153', 'T1151', 'T1193', 'T1192', 'T1194',
                'T1184', 'T1071', 'T1032', 'T1095', 'T1165', 'T1169', 'T1206', 'T1195', 'T1019', 'T1082', 'T1016',
                'T1049', 'T1033', 'T1007', 'T1124', 'T1080', 'T1221', 'T1072', 'T1209', 'T1099', 'T1154', 'T1127',
                'T1199', 'T1111', 'T1065', 'T1204', 'T1078', 'T1125', 'T1102', 'T1100', 'T1077', 'T1047', 'T1084',
                'T1028', 'T1004', 'T1220']
    tech_url_prefix = 'https://attack.mitre.org/techniques/'
    for tech_id in tech_ids:
        tech_id = 'T1091'
        tech_url = tech_url_prefix + tech_id + '/'
        driver.get(tech_url)
        sleep(2)
        scroll_end(driver, lang)
        root = driver.find_elements_by_class_name('container-fluid')[1]
        # relation between tatic and techique
        if lang == 'en':
            tatics = root.find_elements_by_class_name('card-data')[2].text
            tatics = map(lambda x: x.strip(), tatics[len('Tactic: '):].split(','))
            for tatic in tatics:
                tatic_id = select_mysql('mitre_tactics', 'tid', 'name_en', [tatic])[0][0]
                insert_mysql('mitre_rel_tactic_tech', 'tact_id,tech_id', [tatic_id, tech_id])

        # related groups
        groups_e = root.find_element_by_xpath('//tbody[@class="bg-white"]')
        for tr in groups_e.find_elements_by_xpath('tr'):
            exam_id = tr.find_element_by_xpath('td/a[@href]').get_attribute('href').split('/')[-1]
            desc = tr.find_element_by_xpath('td/p').text.split('[')[0]
            # print(exam_id, desc)
            if exam_id.startswith('S'):
                # relation between technique and Software
                if lang == 'en':
                    insert_mysql('mitre_rel_tech_software', 'tid,sid,desc_en', [tech_id, exam_id, desc])
                else:
                    update_mysql('mitre_rel_tech_software', 'desc_cn', [desc], 'tid,sid', [tech_id, exam_id])
            elif exam_id.startswith('G'):
                # relation between technique and Group
                if lang == 'en':
                    insert_mysql('mitre_rel_tech_group', 'tid,gid,desc_en', [tech_id, exam_id, desc])
                else:
                    update_mysql('mitre_rel_tech_group', 'desc_cn', [desc], 'tid,gid', [tech_id, exam_id])

        # technique name
        name = root.find_element_by_tag_name('h1').text.strip()
        # technique name
        desc = root.find_element_by_xpath('div/div').text

        print(len(root.find_elements()))
        # root.find_element_by_xpath('h2/../div').text

        sys.exit()
        a = driver.find_elements_by_tag_name('h2')
        for _a in a:
            print(_a.text)
        break

        element = driver.find_element_by_xpath("//tbody[@class='bg-white']")
        for tr in element.find_elements_by_tag_name('tr'):
            value = list(map(lambda td: td.text, tr.find_elements_by_tag_name('td')))
            event_tids.append(value[0])
            value = value[:2] + [event_tid]
            try:
                if lang == 'cn':
                    sql_str = 'update {} set name_cn="{}" where tid="{}"'.format(tb, value[1].strip('的'), value[0])
                    cur.execute(sql_str)
                else:
                    fields = 'tid, name_en, attack_id'
                    sql_str = 'insert into {}({}) values({})'.format(tb, fields,
                                                                     ','.join(['%s'] * len(fields.split(','))))
                    cur.execute(sql_str, value)
                conn.commit()
            except:
                print(traceback.format_exc())
    driver.quit()
    return event_tids


if __name__ == '__main__':
    # attack_tids = get_tactics(options, 'en')
    # attack_tids = get_tactics(options, 'cn')
    # tech_ids = get_tech_ids()
    # event_tids = get_techniques(None, options, 'en')
    # event_tids = get_techniques(None, options, 'cn')

    get_techniques(None, 'en')
    get_techniques(None, 'cn')
