#!/usr/bin/python
# coding=utf-8

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import re
import sys
import os
import pymysql
from configparser import RawConfigParser
import traceback
import hashlib
import logging.handlers

TACTIC_COUNT = 40
TECHNIQUE_COUNT = 485
GROUP_COUNT = 86
SOFTWARE_COUNT = 377


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

log = logging.getLogger('get_mitre')


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def is_chinese(string):
    for ch in string:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def sub_escape(string):
    pattren1 = re.compile(r' \[\d+\]')
    pattren2 = re.compile(r'\[\d+\]')
    string1 = re.sub(pattren1, '', string)
    string2 = re.sub(pattren2, '', string1)
    return pymysql.escape_string(string2)


def scroll_end(driver, lang='cn', type=''):
    if lang == 'en':
        return
    if type == 'software':
        height = 4000
    else:
        height = driver.execute_script('return document.body.scrollHeight')
    step = 500
    scrolled_height = 0
    sleep(3)
    while scrolled_height <= height:
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


def select_mysql_is_not_null(tb, wh_filed=''):
    assert wh_filed != ''
    sql = 'select count(1) from {} where {} is not NULL'.format(tb, wh_filed)
    with DB() as cur:
        cur.execute(sql)
        result = cur.fetchone()
    return result


def select_mysql(tb, sel_fileds, wh_fileds='', wh_values=[]):
    # fields = 'tid, name_en, attack_id'
    if wh_fileds:
        wh_fileds = wh_fileds.split(',')
        wh_str = 'where ' + ' and '.join([str(f) + '="' + str(v) + '"' for f, v in zip(wh_fileds, wh_values)])
    else:
        wh_str = ''

    sql = 'select {} from {} {}'.format(sel_fileds, tb, wh_str)
    with DB() as cur:
        cur.execute(sql)
        result = cur.fetchall()
    return result


def insert_mysql(tb, fields, values, many_flag=0):
    # fields = 'tid,name_en,attack_id'
    sql = 'insert into {}({}) values({})'.format(tb, fields, ','.join(['%s'] * len(values)))
    with DB() as cur:
        try:
            if many_flag == 0:
                cur.execute(sql, values)
            else:
                cur.executemany(sql, values)
        except Exception as e:
            if e.args[0] == 1062:  # Duplicate entry
                pass
            else:
                log.error(traceback.format_exc())


def update_mysql(tb, set_fields, set_values, wh_fileds, wh_values):
    # fields = 'tid,name_en,attack_id'
    set_fields = set_fields.split(',')
    wh_fileds = wh_fileds.split(',')
    set_str = ','.join([str(f) + '="' + str(v) + '"' for f, v in zip(set_fields, set_values)])
    wh_str = ' and '.join([str(f) + '="' + str(v) + '"' for f, v in zip(wh_fileds, wh_values)])
    sql = 'update {} set {} where {}'.format(tb, set_str, wh_str)
    with DB() as cur:
        cur.execute(sql)


def get_tactics(lang):
    if lang == 'en':
        if TACTIC_COUNT == select_mysql_is_not_null('mitre_tactics', 'desc_en')[0]:
            return
    else:
        if TACTIC_COUNT == select_mysql_is_not_null('mitre_tactics', 'desc_cn')[0]:
            return

    _url_prefix = 'https://attack.mitre.org/tactics/'
    domains = ['pre', 'enterprise', 'mobile']
    driver = webdriver.Chrome(options=get_chrome_option(lang))
    for domain in domains:
        _url = _url_prefix + domain + '/'
        driver.get(_url)
        if lang == 'cn':
            sleep(40)
        scroll_end(driver, lang)
        root = driver.find_element_by_xpath('//tbody[@class="bg-white"]')
        for tr in root.find_elements_by_xpath('tr'):
            tds = tr.find_elements_by_xpath('td')
            tid = tds[0].text
            name = tds[1].text
            desc = pymysql.escape_string(tds[2].text)
            if lang == 'en':
                insert_mysql('mitre_tactics', 'tid,domain,name_en,desc_en', [tid, domain, name, desc])
            else:
                if is_chinese(desc):
                    update_mysql('mitre_tactics', 'name_cn,desc_cn', [name, desc], 'tid', [tid])
                else:
                    log.error('url[%s] not translated successfully', _url)
                    break

    driver.quit()


def get_technique_ids():
    if TECHNIQUE_COUNT == select_mysql('mitre_techniques', 'count(1)')[0][0]:
        return

    _url_prefix = 'https://attack.mitre.org/techniques/'
    domains = ['pre', 'enterprise', 'mobile']
    driver = webdriver.Chrome()
    for domain in domains:
        _url = _url_prefix + domain + '/'
        driver.get(_url)
        element = driver.find_element_by_xpath("//tbody[@class='bg-white']")
        for tr in element.find_elements_by_tag_name('tr'):
            _id = tr.find_element_by_tag_name('td').text
            insert_mysql('mitre_techniques', 'tid,domain', [_id, domain])
    driver.quit()


def get_techniques(lang):
    _ids = sum(select_mysql('mitre_techniques', 'tid'), ())
    flag_txt = 'tech_flag_{}.txt'.format(lang)
    if not os.path.exists(flag_txt):
        open(flag_txt, 'w').close()
    with open(flag_txt) as f:
        processed_ids = list(map(lambda line: line[:-1], f.readlines()))

    driver = webdriver.Chrome(options=get_chrome_option(lang))
    _url_prefix = 'https://attack.mitre.org/techniques/'
    for _id in _ids:
        if _id in processed_ids:
            continue
        log.info('processing technique: %s', _id)

        _url = _url_prefix + _id + '/'
        driver.get(_url)
        scroll_end(driver, lang)
        root = driver.find_elements_by_class_name('container-fluid')[1]

        # technique name
        name = root.find_element_by_tag_name('h1').text.strip()
        # technique name
        desc = root.find_element_by_xpath('div/div').text
        desc = sub_escape(desc)

        if lang == 'cn' and not is_chinese(desc):
            log.error('url[%s] not translated successfully', _url)
            continue

        # relation between techique and tatic
        if lang == 'en':
            tatics = root.find_elements_by_class_name('card-data')[2].text
            tatics = map(lambda x: x.strip(), tatics[len('Tactic: '):].split(','))
            for tatic in tatics:
                result = select_mysql('mitre_tactics', 'tid', 'name_en', [tatic])
                if result:  # else---*Deprecation Warning****
                    tatic_id = result[0][0]
                    insert_mysql('mitre_rel_tech_tactic', 'tech_id,tact_id', [_id, tatic_id])

        # related groups
        try:
            table_e = root.find_element_by_xpath('//tbody[@class="bg-white"]')
            for tr in table_e.find_elements_by_xpath('tr'):
                exam_id = tr.find_element_by_xpath('td/a[@href]').get_attribute('href').split('/')[-1]
                desc = tr.find_element_by_xpath('td/p').text
                desc = sub_escape(desc)

                if exam_id.startswith('S'):
                    # relation between technique and Software
                    if lang == 'en':
                        insert_mysql('mitre_rel_tech_software', 'tid,sid,desc_en', [_id, exam_id, desc])
                    else:
                        update_mysql('mitre_rel_tech_software', 'desc_cn', [desc], 'tid,sid', [_id, exam_id])
                elif exam_id.startswith('G'):
                    # relation between technique and Group
                    if lang == 'en':
                        insert_mysql('mitre_rel_tech_group', 'tid,gid,desc_en', [_id, exam_id, desc])
                    else:
                        update_mysql('mitre_rel_tech_group', 'desc_cn', [desc], 'tid,gid', [_id, exam_id])
        except NoSuchElementException:
            # T1006 T1013 T1042 T1051 T1054 T1058 T1062 T1118 T1121 T1139 T1143 T1144 T1146 T1147 T1148 T1149 T1150...
            log.info('technique[%s] has no example.', _id)

        miti_data = ''
        dete_data = ''
        try:
            # get Mitigation
            miti = root.find_element_by_xpath("h2[@id='mitigation']/following::*")
            while miti.tag_name != 'h2':
                miti_data = miti_data + miti.text + '\n'
                miti = miti.find_element_by_xpath('following::*')
            miti_data = sub_escape(miti_data.strip())

            # get Detection
            dete = root.find_element_by_xpath("h2[@id='detection']/following::*")
            while dete.tag_name != 'h2':
                dete_data = dete_data + dete.text + '\n'
                dete = dete.find_element_by_xpath('following::*')
            dete_data = sub_escape(dete_data.strip())
        except NoSuchElementException:
            pass

        if lang == 'en':
            refs_e = driver.find_elements_by_class_name('scite-citation-text')
            for ref_e in refs_e:
                ref = ref_e.text
                md5 = hashlib.md5()
                md5.update((ref + _id).encode('utf8'))
                rid = md5.hexdigest()

                try:
                    url = ref_e.find_element_by_tag_name('a').get_attribute('href')
                except NoSuchElementException:
                    url = ''
                    log.info('ref[%s] has no url.', rid)
                insert_mysql('mitre_references', 'rid,ref,url,technique_id', [rid, ref, url, _id])

        if lang == 'en':
            update_mysql('mitre_techniques', 'name_en,desc_en,mitigation_en,detection_en',
                         [name, desc, miti_data, dete_data], 'tid', [_id])
        else:
            update_mysql('mitre_techniques', 'name_cn,desc_cn,mitigation_cn,detection_cn',
                         [name, desc, miti_data, dete_data], 'tid', [_id])

        with open(flag_txt, 'a') as f:
            f.write(_id + '\n')
    driver.quit()


def get_group_ids(lang):
    if lang == 'en':
        if GROUP_COUNT == select_mysql_is_not_null('mitre_groups', 'desc_en')[0]:
            return
    else:
        if GROUP_COUNT == select_mysql_is_not_null('mitre_groups', 'desc_cn')[0]:
            return

    base_url = 'https://attack.mitre.org/groups/'
    driver = webdriver.Chrome(options=get_chrome_option(lang))
    driver.get(base_url)
    if lang == 'cn':
        sleep(40)
    scroll_end(driver, lang)
    element = driver.find_element_by_xpath("//tbody[@class='bg-white']")
    for tr in element.find_elements_by_tag_name('tr'):
        tds = tr.find_elements_by_tag_name('td')
        gid = tds[0].find_element_by_tag_name('a').get_attribute('href').split('/')[-2]
        name = tds[0].text
        alias = tds[1].text
        desc = tds[2].text
        if lang == 'en':
            insert_mysql('mitre_groups', 'gid,name,alias,desc_en', [gid, name, alias, desc])
        else:
            if is_chinese(desc):
                update_mysql('mitre_groups', 'desc_cn', [desc], 'gid', [gid])
            else:
                log.error('url[%s] not translated successfully', base_url)
                break

    driver.quit()


def get_groups(lang):
    _ids = sum(select_mysql('mitre_groups', 'gid'), ())
    flag_txt = 'group_flag_{}.txt'.format(lang)
    if not os.path.exists(flag_txt):
        open(flag_txt, 'w').close()
    with open(flag_txt) as f:
        processed_ids = list(map(lambda line: line[:-1], f.readlines()))

    driver = webdriver.Chrome(options=get_chrome_option(lang))

    _url_prefix = 'https://attack.mitre.org/groups/'
    for _id in _ids:
        if _id in processed_ids:
            continue
        log.info('processing group: %s', _id)
        _url = _url_prefix + _id + '/'
        driver.get(_url)
        scroll_end(driver, lang)
        root = driver.find_elements_by_class_name('container-fluid')[1]

        desc = root.find_element_by_xpath('div/div').text
        desc = sub_escape(desc)
        if lang == 'cn' and not is_chinese(desc):
            log.error('url[%s] not translated successfully', _url)
            continue

        # relation between group and technique
        try:
            table_e = root.find_element_by_xpath("h2[@id='techniques']/following::*").find_element_by_xpath(
                'tbody[@class="bg-white"]')
            for tr in table_e.find_elements_by_xpath('tr'):
                tds = tr.find_elements_by_xpath('td')
                domain = tds[0].text
                tid = tds[1].text
                name = tds[2].text
                use = sub_escape(tds[3].text)

                if lang == 'en':
                    insert_mysql('mitre_rel_group_tech', 'group_id,tech_id,domain,name_en,use_en',
                                 [_id, tid, domain, name, use])
                else:
                    update_mysql('mitre_rel_group_tech', 'name_cn,use_cn', [name, use], 'group_id,tech_id', [_id, tid])
        except NoSuchElementException:
            log.info('group[%s] has no technique.', _id)

        # relation between group and software
        try:
            table_e = root.find_element_by_xpath("h2[@id='software']/following::*").find_element_by_xpath(
                'tbody[@class="bg-white"]')
            for tr in table_e.find_elements_by_xpath('tr'):
                tds = tr.find_elements_by_xpath('td')
                sid = tds[0].text
                name = tds[1].text
                techniques = tds[-1].text
                if lang == 'en':
                    insert_mysql('mitre_rel_group_software', 'group_id,software_id,name,techniques',
                                 [_id, sid, name, techniques])
        except NoSuchElementException:
            log.info('group[%s] has no software.', _id)

        # relation between group and reference
        if lang == 'en':
            refs_e = driver.find_elements_by_class_name('scite-citation-text')
            for ref_e in refs_e:
                ref = ref_e.text
                md5 = hashlib.md5()
                md5.update((ref + _id).encode('utf8'))
                rid = md5.hexdigest()

                try:
                    url = ref_e.find_element_by_tag_name('a').get_attribute('href')
                except NoSuchElementException:
                    url = ''
                    log.info('ref[%s] has no url.', rid)
                insert_mysql('mitre_references', 'rid,ref,url,group_id', [rid, ref, url, _id])

        with open(flag_txt, 'a') as f:
            f.write(_id + '\n')
    driver.quit()


def get_software_ids(lang):
    if lang == 'en':
        if SOFTWARE_COUNT == select_mysql_is_not_null('mitre_softwares', 'desc_en')[0]:
            return
    else:
        if SOFTWARE_COUNT == select_mysql_is_not_null('mitre_softwares', 'desc_cn')[0]:
            return

    base_url = 'https://attack.mitre.org/software/'
    driver = webdriver.Chrome(options=get_chrome_option(lang))
    driver.get(base_url)
    if lang == 'cn':
        sleep(40)
    scroll_end(driver, lang)
    element = driver.find_element_by_xpath("//tbody[@class='bg-white']")
    for tr in element.find_elements_by_tag_name('tr'):
        tds = tr.find_elements_by_tag_name('td')
        sid = tds[0].find_element_by_tag_name('a').get_attribute('href').split('/')[-1]
        name = tds[0].text
        associated_software = tds[1].text
        desc = tds[2].text
        if lang == 'en':
            insert_mysql('mitre_softwares', 'sid,name,associated_software,desc_en',
                         [sid, name, associated_software, desc])
        else:
            if is_chinese(desc):
                update_mysql('mitre_softwares', 'desc_cn', [desc], 'sid', [sid])
            else:
                log.error('url[%s] not translated successfully', base_url)
                break
    driver.quit()


def get_softwares(lang):
    _ids = sum(select_mysql('mitre_softwares', 'sid'), ())
    flag_txt = 'software_flag_{}.txt'.format(lang)
    if not os.path.exists(flag_txt):
        open(flag_txt, 'w').close()
    with open(flag_txt) as f:
        processed_ids = list(map(lambda line: line[:-1], f.readlines()))

    driver = webdriver.Chrome(options=get_chrome_option(lang))

    _url_prefix = 'https://attack.mitre.org/software/'
    for _id in _ids:
        if _id in processed_ids:
            continue
        log.info('processing software: %s', _id)
        _url = _url_prefix + _id + '/'
        driver.get(_url)
        scroll_end(driver, lang, 'software')
        root = driver.find_elements_by_class_name('container-fluid')[1]

        desc = root.find_element_by_xpath('div/div').text
        desc = sub_escape(desc)
        if lang == 'cn' and not is_chinese(desc):
            log.error('url[%s] not translated successfully', _url)
            continue

        # relation between software and technique
        try:
            table_e = root.find_element_by_xpath("h2[@id='techniques']/following::*").find_element_by_xpath(
                'tbody[@class="bg-white"]')
            for tr in table_e.find_elements_by_xpath('tr'):
                tds = tr.find_elements_by_xpath('td')
                domain = tds[0].text
                tid = tds[1].text
                name = tds[2].text
                use = sub_escape(tds[3].text)
                if lang == 'en':
                    insert_mysql('mitre_rel_software_tech', 'software_id,tech_id,domain,name_en,use_en',
                                 [_id, tid, domain, name, use])
                else:
                    update_mysql('mitre_rel_software_tech', 'name_cn,use_cn', [name, use], 'software_id,tech_id',
                                 [_id, tid])
        except NoSuchElementException:
            log.info('software[%s] has no technique.', _id)

        # relation between software and group
        if lang == 'en':
            try:
                group_e = root.find_element_by_xpath("h2[@id='groups']/following::*/following::*")
                while group_e.tag_name == 'a':
                    gid = group_e.get_attribute('href').split('/')[-2]
                    name = group_e.text
                    insert_mysql('mitre_rel_software_group', 'software_id,group_id,name', [_id, gid, name])
                    group_e = group_e.find_element_by_xpath('following::*/following::*')
            except NoSuchElementException:
                log.info('software[%s] has no group.', _id)

        # relation between software and reference
        if lang == 'en':
            refs_e = driver.find_elements_by_class_name('scite-citation-text')
            for ref_e in refs_e:
                ref = ref_e.text
                md5 = hashlib.md5()
                md5.update((ref + _id).encode('utf8'))
                rid = md5.hexdigest()

                try:
                    url = ref_e.find_element_by_tag_name('a').get_attribute('href')
                except NoSuchElementException:
                    url = ''
                    log.info('ref[%s] has no url.', rid)
                insert_mysql('mitre_references', 'rid,ref,url,software_id', [rid, ref, url, _id])

        with open(flag_txt, 'a') as f:
            f.write(_id + '\n')
    driver.quit()


if __name__ == '__main__':
    init_logging('get_mitre.log')
    langs = ['en', 'cn']
    for lang in langs:
        get_tactics(lang)
        get_technique_ids()
        get_techniques(lang)
        get_group_ids(lang)
        get_groups(lang)
        get_software_ids(lang)
        get_softwares(lang)
