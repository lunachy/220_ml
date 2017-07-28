# coding=utf-8
import sys
from datetime import datetime, timedelta
from time import sleep
from collections import defaultdict
from pysolr import Solr
import MySQLdb

# from solr import SolrConnection
aps = ['Intelligence\ Gathering', 'Point\ of\ Entry', 'Command\ and\ Control\ Communication',
       'Lateral\ Movement', 'Asset\ and\ Data\ Discovery', 'Data\ Exfiltration']


def cal_attacks(ip_addr, today):
    tda_solr = Solr('{0}/solr/tda_{1}'.format(ip_addr, today))
    waf_solr = Solr('{0}/solr/waf_{1}'.format(ip_addr, today))
    ids_solr = Solr('{0}/solr/ids_{1}'.format(ip_addr, today))

    threat_types_result = tda_solr.search('*:*', facet='on', **{'facet.field': ['threatType_s']})
    # [u'1', 25662, u'2', 16844, u'4', 3503, u'0', 639, u'3', 29]
    values = threat_types_result.facets['facet_fields']['threatType_s']
    threat_types = defaultdict(lambda: 0, zip(values[0::2], values[1::2]))

    waf_attacks = waf_solr.search('*:*').hits
    tda_attacks = tda_solr.search('*:*').hits
    ids_attacks = ids_solr.search('*:*').hits
    all_attacks = tda_attacks + waf_attacks + ids_attacks

    # 0: Malicious content
    # 1: Malicious behavior
    # 2: Suspicious behavior
    # 3: Exploit
    # 4: Grayware
    # 5: Malicious URLs
    # 6: Disruptive Applications
    # 7: 全攻击量:waf + TDA + IDS攻击量
    # 8: WAF攻击量
    # 9: TDA攻击量
    # 10:IDS攻击量
    values = zip([today] * 11, [str(i) for i in range(11)],
                 [threat_types[str(i)] for i in range(7)] + [all_attacks, waf_attacks, tda_attacks, ids_attacks])

    conn = MySQLdb.connect(host='10.15.42.21', port=3306, user='root', passwd='root', db='traffic')
    cur = conn.cursor()
    cur.executemany('insert into predict(ThreatDate, ThreatType, ThreatNumber) values(%s, %s, %s)', values)
    conn.commit()
    cur.close()
    conn.close()


# start = 0
# all_results = []
# solr = Solr("http://10.21.17.209:8181/solr/tda_20170614")
# solr.delete(q='*:*')
# while True:
#     results = solr.search('pattackPhase_s:*', start=start, rows=100)
#     start += len(results)
#     if len(results) == 0:
#         break
#     all_results += results.docs
# print start, len(all_results)
#
# for i, result in enumerate(all_results):
#     print result['pattackPhase_s']
#     if i > 3:
#         break
# sys.exit()


def update_uuid1(id0, uuid1):
    doc = [{"id": id0, "RelationUUID_s": uuid1}]
    tda_solr.add(doc, fieldUpdates={"RelationUUID_s": "set"})


def update_uuid(r0, uuid1):
    # TODO: limit time interval
    if r0['RelationUUID_s'] != '0':
        return

    doc = [{"id": r0['id'], "RelationUUID_s": uuid1}]
    tda_solr.add(doc, fieldUpdates={"RelationUUID_s": "set"})


def set_pattack(tda_solr, value='0', attack_phase=aps[1]):
    results = tda_solr.search('pattackPhase_s:{}'.format(attack_phase), rows=10000)
    for r in results:
        doc = [{"id": r['id'], "RelationUUID_s": value}]
        tda_solr.add(doc, fieldUpdates={"RelationUUID_s": "set"})


def solr_search(ap, direction, RelationUUID=0):
    fq = ['pattackPhase_s:{}'.format(ap), 'direction_s:{}'.format(direction)]
    if RelationUUID == 0:
        fq.append('RelationUUID_s:0')
    results = tda_solr.search('*:*', fq=fq, rows=10000)
    # print ap, results.hits
    return results


def apt_relate(tda_solr):
    start = 0
    # results0 = []
    # results1 = []
    # for i, ap in enumerate(aps):
    #     if i == 1:  # 'Point\ of\ Entry'
    #         fq0 = ['pattackPhase_s:{}'.format(ap), 'direction_s:0']
    #         fq1 = ['pattackPhase_s:{}'.format(ap), 'direction_s:1']
    #     else:
    #         fq0 = ['pattackPhase_s:{}'.format(ap), 'direction_s:0', 'RelationUUID_s:0']
    #         fq1 = ['pattackPhase_s:{}'.format(ap), 'direction_s:1', 'RelationUUID_s:0']
    #
    #     r0 = tda_solr.search('*:*', fq=fq0, rows=10000)
    #     r1 = tda_solr.search('*:*', fq=fq1, rows=10000)
    #     print ap, r0.hits, r1.hits
    #     results0.append(r0)
    #     results1.append(r1)

    # print map(lambda x: len(x), results0), map(lambda x: len(x), results1)
    # outer---point of entry
    # for r01 in results0[1]:
    for r01 in solr_search(aps[1], 0, 1):
        uuid1 = r01['RelationUUID_s']
        # phase0 < --->phase1
        for r00 in solr_search(aps[0], 0):  # results0[0]:
            if r00['dstIp_s'] == r01['dstIp_s']:
                update_uuid(r00, uuid1)

        # phase1 < --->phase2
        for r02 in solr_search(aps[2], 0):  # for r02 in results0[2]:
            print r02['dstIp_s'], r01['dstIp_s'], 1, r02['RelationUUID_s'], r01['RelationUUID_s']
            if r02['dstIp_s'] == r01['dstIp_s']:
                update_uuid(r02, uuid1)
                # print r02['id'], r01['RelationUUID_s'], r02['RelationUUID_s']

                # phase2 < --->phase3
                for r13 in solr_search(aps[3], 1):  # for r13 in results1[3]:
                    print r13['dstIp_s'], r02['dstIp_s'], 11, r13['RelationUUID_s'], r02['RelationUUID_s']
                    if r13['dstIp_s'] == r02['dstIp_s']:
                        update_uuid(r13, uuid1)

                        # phase3 < --->phase4
                        for r04 in solr_search(aps[4], 0):  # for r04 in results0[4]:
                            print r04['dstIp_s'], r13['dstIp_s'], 111, r04['RelationUUID_s'], r13['RelationUUID_s']
                            if r04['dstIp_s'] == r13['dstIp_s'] or r04['dstIp_s'] == r13['srcIp_s']:
                                update_uuid(r04, uuid1)

    for r11 in solr_search(aps[1], 1, 1):  # results1[1]:
        uuid1 = r11['RelationUUID_s']
        for r00 in solr_search(aps[0], 0):  # results0[0]:
            if r00['dstIp_s'] == r11['srcIp_s']:
                update_uuid(r00, uuid1)

        for r02 in solr_search(aps[2], 0):  # results0[2]:
            print r02['dstIp_s'], r11['srcIp_s'], 2
            if r02['dstIp_s'] == r11['srcIp_s']:
                update_uuid(r02, uuid1)

                for r13 in solr_search(aps[3], 1):  # results1[3]:
                    if r13['dstIp_s'] == r02['dstIp_s']:
                        update_uuid(r13, uuid1)

                        for r04 in solr_search(aps[4], 0):  # results0[4]:
                            if r04['dstIp_s'] == r13['dstIp_s'] or r04['dstIp_s'] == r13['srcIp_s']:
                                update_uuid(r04, uuid1)

    for r01 in solr_search(aps[1], 0, 1):  # results0[1]:
        uuid1 = r01['RelationUUID_s']
        for r10 in solr_search(aps[0], 1):  # results1[0]:
            if r10['srcIp_s'] == r01['dstIp_s']:
                update_uuid(r10, uuid1)

        for r12 in solr_search(aps[2], 1):  # results1[2]:
            print r12['srcIp_s'], r01['dstIp_s'], 3
            if r12['srcIp_s'] == r01['dstIp_s']:
                update_uuid(r12, uuid1)

                for r13 in solr_search(aps[3], 1):  # results1[3]:
                    if r13['dstIp_s'] == r12['srcIp_s']:
                        update_uuid(r13, uuid1)

                        for r04 in solr_search(aps[4], 0):  # results0[4]:
                            if r04['dstIp_s'] == r13['dstIp_s'] or r04['dstIp_s'] == r13['srcIp_s']:
                                update_uuid(r04, uuid1)

    for r11 in solr_search(aps[1], 1, 1):  # results1[1]:
        uuid1 = r11['RelationUUID_s']
        for r10 in solr_search(aps[0], 1):  # results1[0]:
            if r10['srcIp_s'] == r11['srcIp_s']:
                update_uuid(r10, uuid1)

        for r12 in solr_search(aps[2], 1):  # results1[2]:
            print r12['srcIp_s'], r11['srcIp_s'], 4
            if r12['srcIp_s'] == r11['srcIp_s']:
                update_uuid(r12, uuid1)

                for r13 in solr_search(aps[3], 1):  # results1[3]:
                    if r13['dstIp_s'] == r12['srcIp_s']:
                        update_uuid(r13, uuid1)

                        for r04 in solr_search(aps[4], 0):  # results0[4]:
                            if r04['dstIp_s'] == r13['dstIp_s'] or r04['dstIp_s'] == r13['srcIp_s']:
                                update_uuid(r04, uuid1)

    while True:
        uuid0_results = tda_solr.search('RelationUUID_s:0', start=start, rows=1000)
        start += len(uuid0_results)
        if len(uuid0_results) == 0:
            break
        for r in uuid0_results:
            if (datetime.now() - datetime.strptime(r['logTime_dt'],
                                                   "%Y-%m-%dT%H:%M:%SZ")).seconds > 24 * 60 * 60:
                update_uuid1(r['id'], '1')


if __name__ == '__main__':
    ip_addr = 'http://10.21.17.207:8181'
    time_now = datetime.now()
    today = datetime.now().strftime('%Y%m%d')
    tda_solr = Solr('{0}/solr/tda_{1}'.format(ip_addr, today))
    # tda_solr.delete(q='*:*')
    # set_pattack(tda_solr)
    apt_relate(tda_solr)

    # while True:
    #     try:
    #         time_now = datetime.now()
    #         today = time_now.strftime('%Y%m%d')
    #         yesterday = (time_now - timedelta(days=1)).strftime('%Y%m%d')
    #         if time_now.hour == 3:
    #             cal_attacks(ip_addr, yesterday)
    #         tda_solr = Solr('{0}/solr/tda_{1}'.format(ip_addr, today))
    #         apt_relate(tda_solr)
    #         sleep(1 * 60 * 60)
    #     except Exception as e:
    #         print e
