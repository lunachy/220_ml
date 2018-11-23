# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python_package'))
import time
import signal
import logging.handlers
from ConfigParser import RawConfigParser
import pymysql
from multiprocessing import Pool, cpu_count, Value
import requests
from random import choice
import traceback
import argparse
from datetime import datetime, timedelta

log = logging.getLogger(__file__[:-3])
CPU_COUNT = cpu_count()
KEYS = ['2f953c1436cb4803ed92e2897d644f5a3ffeed2d3ee25ff8a99c8ef402414193',
        '7f3b87f683a302eab4c5b49d2765987587aedbd6b76c22c912ebc2af67ff1911',
        'f3cc37e4ca64c62a1df4c5db7c9296dd928a1d7f39c78c88e74829f70e66e15b',
        '863dbdd0b90bf0f2471efeac9bc2dad5e8ee7002b2f6746da6ddd7b8059a1ed3',
        'beb30ccc839cb6c10840b4002263992fe0b3ec1da109c984983f4360283b5764',
        'f677ec5ae0f83ce403b6d340639779581a6f4fa1e273fd9040e9c331a705be1f',
        'e78e0705c8f9e927c2684570bea5782df75ac3c1fd7cc6442d9a1176d9d854c2',
        '20e67d281302544f296f0edf434531ddebec08fa813971624bd46a3876fdbe42',
        'e598222fea48b4e6c6ba3c9fb32376901dad50c3972b5f21b8130d7f705d25f7',
        'a23a8a5c33a5ddd5caad0617fd1c7e18922c910698bf598b666aee167c8f7c08',
        '4a5cd77eb7115589429cb51c1af8fc729b1040225e3fa0415070ce20cd8274bf',
        'f7b31dea8b2661cb69a39669875f3ebfd32d693590095d87bbb0a7cf67c6515a',
        '6468d26f08ce199c189ef57e17c3d3764864999f7ed544e71d34668bd4a89bc9',
        '908e8371f2a843dcccbbd25b249cedf3833380b89b9a71b6b24949aa3ead62ac',
        'f036345feaee8cf4c8ace77bae629288d661b8ec689ac38cf6c969dca6a266ad',
        '73e86529b50ce220b319ffdd2d69995f98d6be5ebb3b4526fe0f7bb30d08e0e7',
        '2f953c1436cb4803ed92e2897d644f5a3ffeed2d3ee25ff8a99c8ef402414193',
        '7f3b87f683a302eab4c5b49d2765987587aedbd6b76c22c912ebc2af67ff1911',
        'f3cc37e4ca64c62a1df4c5db7c9296dd928a1d7f39c78c88e74829f70e66e15b',
        '863dbdd0b90bf0f2471efeac9bc2dad5e8ee7002b2f6746da6ddd7b8059a1ed3',
        'beb30ccc839cb6c10840b4002263992fe0b3ec1da109c984983f4360283b5764',
        'f677ec5ae0f83ce403b6d340639779581a6f4fa1e273fd9040e9c331a705be1f',
        'e78e0705c8f9e927c2684570bea5782df75ac3c1fd7cc6442d9a1176d9d854c2',
        '20e67d281302544f296f0edf434531ddebec08fa813971624bd46a3876fdbe42',
        'e598222fea48b4e6c6ba3c9fb32376901dad50c3972b5f21b8130d7f705d25f7',
        'a23a8a5c33a5ddd5caad0617fd1c7e18922c910698bf598b666aee167c8f7c08',
        '4a5cd77eb7115589429cb51c1af8fc729b1040225e3fa0415070ce20cd8274bf',
        'f7b31dea8b2661cb69a39669875f3ebfd32d693590095d87bbb0a7cf67c6515a',
        '6468d26f08ce199c189ef57e17c3d3764864999f7ed544e71d34668bd4a89bc9',
        '908e8371f2a843dcccbbd25b249cedf3833380b89b9a71b6b24949aa3ead62ac',
        'f036345feaee8cf4c8ace77bae629288d661b8ec689ac38cf6c969dca6a266ad',
        '73e86529b50ce220b319ffdd2d69995f98d6be5ebb3b4526fe0f7bb30d08e0e7',
        '7e8b3f9c0984a4a45ec083f95cff0050cd1997d257e7ff8481794d03c1063440',
        '66da70fc8725d8f573ac7b44243133910898687d7ec9ad5cb53fff5fdf0938a1',
        '8aa9d5430332430876a6e96dff8dd33a898ff550232629029d83f001ef280732',
        '1917ee018de54e554aa83e8906f92b481c56c305fb13e3f1e930b42e9b3ae20a',
        'e7828165a3039bcf92854f56c88a440b267eaec3fb0b1724ebd19e8f19a621ba',
        '8be7b92aed93a23c0884cc090e6f7432cf6146ca8d4f97288bfa85062e57ecfb',
        '5020495d31993b085126f0e25048334487bf0e83e448de1a85ba67b472147937',
        '3b2c2c5a208a8bce59de488beabc933b3baef019d5a1d392646507868a36a83e',
        'c415592b4c10fce8dedac2620a210684561888aa59d079b61cd93c8d5274ca75',
        '227894922a98fcb38fc54d5bcb8da8591aa38970192ede052eee8b0308120896',
        'ee5183c3d1a44d00f379721572e611e633e38d4dab3c6038b1dd3a0ea09faf6c',
        '0b2032a8a1bb3e0b3204f8f7b26ca681052cf2cf5c55cc2850617a9229b5085f',
        '11d8d4b89eb6a86f26327b57615d20b5bbad274487da641879cf52583571493e',
        '44f1dbf461b168ece3caa9d0fb7295a912331bc5676fc94ac7e11f2338764f84',
        'd0d9235f1d156db8070f4e48eb6316d737b00b8d2b1638e98fd61929077901df',
        '75753cb31a239408eaff0789c0023cfb478d489cd5e053a5866f1b2eaef14953',
        '5703e745d8a1edd4a8b4796d0754b29d6a1dc03f9a7a7fea4a8bc0aafffb3d31',
        'ff1745538ce2270fb5ceb2e3706857e9f036d746c780ef8b8e1a2fecbc488a4d',
        'dcde3d700e8c3671bea1d1391c459b9c39a38a471e0545325b531d20c08006cd',
        '240b9158e3a79caa46cea2d17ea137ed8268ff6913f41027f53f548b99ff22c6',
        '3864f324f321240ed75fc1e1c0a969ef485a10ee94d73c92acf0b243f503894a',
        '176f0b1b97bffa716559129ab878495a783caa72e216b837f7aa8f28413515b1',
        '00f553df406b8cedcafb504c34583908f4d1d15f1e09d135b75bf018e0de07b5',
        'c10bb813b503ea321031e4fb893a5d3903beca7cd26aa3f6096f285d1315e312',
        '6403ca4859e036b66231c16984f9a1051242e9fdcc311f65f8c8a2395229a2a4',
        'e9feb958870d592157284bad190e2ed67f525b34c1ebac149fe888c685d5fa41',
        'abfb119c77216272f6909667608cbcfdb6e569ee1178e588c446f39460b183c1',
        '55120838f82de4a041382ffefdeb6b7accac770db1c30edbc76a1cff9418b642',
        '1f6d0b2aa12cdc6ea3ef9fc4e95483e0432d3eaaffb1be468d48fa8d4425805c',
        'd5cec9ffa6a0a5cb2e0778af96cf783de99c446cbc79d159ff5697b0b7860d5c',
        'd66d2fca892d08c49db5a569afd824ed4ecb568fc38d0921c7bd9630caa1a06b', ]
KEY_LEN = len(KEYS)

MD5_ENGINES = [u'Bkav', u'MicroWorld-eScan', u'CMC', u'CAT-QuickHeal', u'ALYac', u'Malwarebytes', u'VIPRE',
               u'TheHacker', u'K7GW', u'K7AntiVirus', u'TrendMicro', u'Baidu', u'F-Prot', u'Symantec', u'TotalDefense',
               u'TrendMicro-HouseCall', u'Paloalto', u'ClamAV', u'GData', u'Kaspersky', u'BitDefender',
               u'NANO-Antivirus', u'ViRobot', u'SUPERAntiSpyware', u'Rising', u'Ad-Aware', u'Sophos', u'Comodo',
               u'F-Secure', u'DrWeb', u'Zillya', u'Invincea', u'McAfee-GW-Edition', u'Emsisoft', u'Ikarus', u'Cyren',
               u'Jiangmin', u'Webroot', u'Avira', u'Antiy-AVL', u'Kingsoft', u'Endgame', u'Arcabit', u'AegisLab',
               u'ZoneAlarm', u'Avast-Mobile', u'Microsoft', u'AhnLab-V3', u'McAfee', u'AVware', u'MAX', u'VBA32',
               u'Cylance', u'WhiteArmor', u'Panda', u'Zoner', u'ESET-NOD32', u'Tencent', u'Yandex', u'SentinelOne',
               u'eGambit', u'Fortinet', u'AVG', u'Cybereason', u'Avast', u'CrowdStrike', u'Qihoo-360']

URL_ENGINES = [u'CLEAN MX', u'DNS8', u'VX Vault', u'ZDB Zeus', u'Tencent', u'AutoShun', u'Netcraft', u'PhishLabs',
               u'Zerofox', u'K7AntiVirus', u'Virusdie External Site Scan', u'Quttera', u'AegisLab WebGuard',
               u'MalwareDomainList', u'ZeusTracker', u'zvelo', u'Google Safebrowsing', u'Kaspersky', u'BitDefender',
               u'Dr.Web', u'Certly', u'G-Data', u'C-SIRT', u'CyberCrime', u'Malware Domain Blocklist', u'MalwarePatrol',
               u'Webutation', u'Trustwave', u'Web Security Guard', u'CyRadar', u'desenmascara.me', u'ADMINUSLabs',
               u'Malwarebytes hpHosts', u'Opera', u'AlienVault', u'Emsisoft', u'Malc0de Database', u'SpyEyeTracker',
               u'malwares.com URL checker', u'Phishtank', u'Malwared', u'Avira', u'NotMining', u'OpenPhish',
               u'Antiy-AVL', u'Forcepoint ThreatSeeker', u'FraudSense', u'Comodo Site Inspector', u'Malekal', u'ESET',
               u'Sophos', u'Yandex Safebrowsing', u'SecureBrain', u'Nucleon', u'Sucuri SiteCheck', u'Blueliv',
               u'ZCloudsec', u'SCUMWARE.org', u'ThreatHive', u'FraudScore', u'Rising', u'URLQuery', u'StopBadware',
               u'Fortinet', u'ZeroCERT', u'Spam404', u'securolytics', u'Baidu-International']

IP_CATS = ['XtremeRAT', 'DarkComet', 'Citadel', 'njRAT', 'vSkimmer', 'NetBus', 'Dexter', 'BlackPOS', 'Poison',
           'BlackShades', 'Alina']

IBM_AUTH = 'Basic MGI4ZjZmYjAtZDljNi00Y2VhLThiMWEtODA0Y2MxMzBjODFhOjkzOTY0YTAxLTUzOTItNGQzNS05ZmFkLTRiNDExNjFmNTQwMA=='


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def get_domains(tb):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = "select distinct domain from {} where category_vt is NULL LIMIT {}".format(tb, KEY_LEN)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def get_md5s(tb):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select md5 from {} where flag is Null LIMIT {}'.format(tb, KEY_LEN)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def get_urls(tb):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select distinct url from {} where flag is Null LIMIT {}'.format(tb, KEY_LEN)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def get_ips(tb):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select distinct ip from {} where category_vt is Null LIMIT {}'.format(tb, KEY_LEN)
    cur.execute(sql)
    _r = cur.fetchall()
    cur.close()
    conn.close()
    return _r


def craw_file_single(md5_tuple):
    url = 'https://www.virustotal.com/vtapi/v2/file/report'
    md5 = md5_tuple[0]
    key_i.acquire()
    params = {'apikey': KEYS[key_i.value % KEY_LEN], 'resource': md5}
    key_i.value += 1
    key_i.release()
    try:
        response = requests.get(url, params=params)
        log.info('%s: %s' % (md5, response.status_code))
        if response.status_code == 200:
            _report = response.json()
            update_list = ["`flag`='1'"]
            if _report['response_code'] == 1:
                _scan = _report['scans']
                for _engine in MD5_ENGINES:
                    if _engine in _scan and _scan[_engine]['detected']:
                        update_list.append("`{}`='{}'".format(_engine, string_clean(_scan[_engine]['result'])))
            sql = 'update {} set {} where md5="{}"'.format(tb_file, ','.join(update_list), md5)
            return sql
    except:
        log.error('insert data failed, msg: %s', traceback.format_exc())


def craw_domain_single(domain_tuple):
    url = 'https://www.virustotal.com/vtapi/v2/domain/report'
    domain = domain_tuple[0]
    key_i.acquire()
    params = {'apikey': KEYS[key_i.value % KEY_LEN], 'domain': domain}
    key_i.value += 1
    key_i.release()
    try:
        response = requests.get(url, params=params)
        log.info('%s: %s' % (domain, response.status_code))
        if response.status_code == 200:
            _report = response.json()
            if 'categories' in _report:
                _c = _report['categories'][0]
            elif 'Forcepoint ThreatSeeker category' in _report:
                _c = _report['Forcepoint ThreatSeeker category']
            else:
                _c = 'uncategorized'
            sql = 'update {} set category_vt="{}" where domain="{}"'.format(tb_domain, _c, domain)
            return sql
    except:
        log.error('insert data failed, msg: %s', traceback.format_exc())


def craw_url_single(url_tuple):
    url = 'https://www.virustotal.com/vtapi/v2/url/report'
    _url = url_tuple[0]
    key_i.acquire()
    params = {'apikey': KEYS[key_i.value % KEY_LEN], 'resource': _url}
    key_i.value += 1
    key_i.release()
    try:
        response = requests.get(url, params=params)
        log.info('%s: %s' % (_url, response.status_code))
        if response.status_code == 200:
            _report = response.json()
            update_list = ["`flag`='1'"]
            if _report['response_code'] == 1:
                _scan = _report['scans']
                for _engine in URL_ENGINES:
                    if _engine in _scan and _scan[_engine]['detected']:
                        update_list.append("`{}`='{}'".format(_engine, _scan[_engine]['result']))
            sql = 'update {} set {} where url="{}"'.format(tb_url, ','.join(update_list), _url)
            return sql
    except:
        log.error('insert data failed, msg: %s', traceback.format_exc())


def craw_ip_single(ip_tuple):
    ip = ip_tuple[0]
    url = 'https://api.xforce.ibmcloud.com/ipr/{}'.format(ip)
    header = {'Authorization': IBM_AUTH}
    try:
        response = requests.get(url, headers=header)
        log.info('%s: %s' % (ip, response.status_code))
        if response.status_code == 200:
            _report = response.json()
            _cat = ','.join(_report['cats'].keys())
            if _cat:
                _score = int(_report['score'] * 10)
            else:
                _score = 0
            update_list = ['`category_vt`="{}"'.format(_cat), '`score`="{}"'.format(_score)]
            sql = 'update {} set {} where ip="{}"'.format(tb_ip, ','.join(update_list), ip)
            return sql
    except:
        log.error('insert data failed, msg: %s', traceback.format_exc())


def multi_process(data, func):
    pool = Pool(processes=CPU_COUNT / 2, initializer=init_worker, maxtasksperchild=400)
    results = pool.map(func, data)
    results = filter(lambda x: x, results)
    pool.close()
    pool.join()
    exec_mysql(results)


def exec_mysql(sql_list):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    for sql in sql_list:
        log.info(sql)
        try:
            cur.execute(sql)
        except:
            log.error('insert data failed, msg: %s', traceback.format_exc())
    conn.commit()
    cur.close()
    conn.close()


def craw_domain(tb_domain):
    url = 'https://www.virustotal.com/vtapi/v2/domain/report'
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select DISTINCT domain from {} where category_vt is Null'.format(tb_domain)
    cur.execute(sql)
    domains = cur.fetchall()

    count_204 = 0
    for i, _domain in enumerate(domains):
        sql = "select category_vt from {} where domain='{}'".format(tb_domain, _domain[0])
        cur.execute(sql)
        results = cur.fetchone()
        if results and results[0]:
            sql = "update {} set category_vt='{}' where domain='{}'".format(tb_domain, results[0], _domain[0])
            log.info(sql)
            cur.execute(sql)
            conn.commit()
        else:
            time.sleep(1)
            params = {'apikey': KEYS[(i * 3) % KEY_LEN], 'domain': _domain[0]}
            try:
                response = requests.get(url, params=params)
                log.info('%s: %s' % (_domain[0], response.status_code))
                if response.status_code == 200:
                    _report = response.json()
                    if 'categories' in _report:
                        _c = _report['categories'][0]
                    elif 'Forcepoint ThreatSeeker category' in _report:
                        _c = _report['Forcepoint ThreatSeeker category']
                    else:
                        _c = 'uncategorized'
                    sql = "update {} set category_vt='{}' where domain='{}'".format(tb_domain, _c, _domain[0])
                    log.info(sql)
                    cur.execute(sql)
                    conn.commit()
                elif response.status_code == 204:
                    count_204 += 1
                    if count_204 > 10:
                        log.info('status_code[204] exceed!')
                        break
                else:
                    continue
            except:
                log.error('insert data failed, msg: %s', traceback.format_exc())

    cur.close()
    conn.close()


def craw_url(tb_url):
    url = 'https://www.virustotal.com/vtapi/v2/url/report'
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select distinct url from {} where flag is Null'.format(tb_url)
    cur.execute(sql)
    urls = cur.fetchall()

    count_204 = 0
    for i, _url in enumerate(urls):
        time.sleep(1)
        params = {'apikey': KEYS[(i * 3 + 1) % KEY_LEN], 'resource': _url[0]}
        try:
            response = requests.get(url, params=params)
            log.info('%s: %s' % (_url[0], response.status_code))
            if response.status_code == 200:
                _report = response.json()
                update_list = ["`flag`='1'"]
                if _report['response_code'] == 1:
                    _scan = _report['scans']
                    for _engine in URL_ENGINES:
                        if _engine in _scan and _scan[_engine]['detected']:
                            update_list.append("`{}`='{}'".format(_engine, _scan[_engine]['result']))
                sql = 'update {} set {} where url="{}"'.format(tb_url, ','.join(update_list), _url[0])
                log.info(sql)
                cur.execute(sql)
                conn.commit()
            elif response.status_code == 204:
                count_204 += 1
                if count_204 > 10:
                    log.info('status_code[204] exceed!')
                    break
            else:
                continue
        except:
            log.error('insert data failed, msg: %s', traceback.format_exc())

    cur.close()
    conn.close()


def craw_ip(tb_ip):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select ip, source from {} where category_vt is Null LIMIT 10'.format(tb_ip)
    cur.execute(sql)
    ips = cur.fetchall()

    for i, (_ip, _src) in enumerate(ips):
        print _ip, _src
        try:
            if _src in IP_CATS:
                update_list = '`category_vt`="{}"'.format(_src)
            else:
                update_list = '`category_vt`=""'
            sql = 'update {} set {} where ip="{}"'.format(tb_ip, update_list, _ip)
            log.info(sql)
            cur.execute(sql)
            conn.commit()
        except:
            log.error('insert data failed, msg: %s', traceback.format_exc())
    cur.close()
    conn.close()


def craw_file(tb_file):
    url = 'https://www.virustotal.com/vtapi/v2/file/report'
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    sql = 'select md5 from {} where flag is Null'.format(tb_file)
    cur.execute(sql)
    md5s = cur.fetchall()

    count_204 = 0
    for i, _md5 in enumerate(md5s):
        time.sleep(1)
        params = {'apikey': KEYS[(i * 3 + 2) % KEY_LEN], 'resource': _md5[0]}
        try:
            response = requests.get(url, params=params)
            log.info('%s: %s' % (_md5[0], response.status_code))
            if response.status_code == 200:
                _report = response.json()
                update_list = ["`flag`='1'"]
                if _report['response_code'] == 1:
                    _scan = _report['scans']
                    for _engine in MD5_ENGINES:
                        if _engine in _scan and _scan[_engine]['detected']:
                            update_list.append("`{}`='{}'".format(_engine, string_clean(_scan[_engine]['result'])))
                sql = "update {} set {} where md5='{}'".format(tb_file, ','.join(update_list), _md5[0])
                log.info(sql)
                cur.execute(sql)
                conn.commit()
            elif response.status_code == 204:
                count_204 += 1
                if count_204 > 10:
                    log.info('status_code[204] exceed!')
                    break
            else:
                continue
        except:
            log.error('insert data failed, msg: %s', traceback.format_exc())

    cur.close()
    conn.close()


def string_clean(string):
    # `Ikarus`='2893592 'Trojan-Dropper.Agent'
    if "'" in string:
        i = string.index("'")
        return string[i + 1:]
    else:
        return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", help="crawl domain info", action="store_true", required=False)
    parser.add_argument("-u", "--url", help="crawl url info", action="store_true", required=False)
    parser.add_argument("-i", "--ip", help="crawl ip info", action="store_true", required=False)
    parser.add_argument("-f", "--file", help="crawl file info", action="store_true", required=False)
    parser.add_argument("--debug", help="debug mode", action="store_true", required=False)
    args = parser.parse_args()

    init_logging(os.path.abspath(__file__)[:-2] + 'log')
    # log.info('start process!')
    key_i = Value('i', 0)
    this_month = datetime.now().strftime('%Y%m')
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    if args.domain:
        tb_domain = 'black_domain'
        craw_domain(tb_domain)
        # domains = get_domains(tb_domain)
        # while domains:
        #     domains = get_domains(tb_domain)
        #     multi_process(domains, craw_domain_single)
        #     time.sleep(15)
    if args.url:
        tb_url = 'black_url'
        craw_url(tb_url)
        # urls = get_urls(tb_url)
        # while urls:
        #     urls = get_urls(tb_url)
        #     multi_process(urls, craw_url_single)
        #     time.sleep(15)
    if args.ip:
        tb_ip = 'black_ip'
        craw_ip(tb_ip)
        # ips = get_ips(tb_ip)
        # while ips:
        #     ips = get_ips(tb_ip)
        #     multi_process(ips, craw_ip_single)
        #     time.sleep(15)
    if args.file:
        tb_file = 'black_file'
        craw_file(tb_file)
        # md5s = get_md5s(tb_file)
        # while md5s:
        #     md5s = get_md5s(tb_file)
        #     multi_process(md5s, craw_file_single)
        #     time.sleep(15)
