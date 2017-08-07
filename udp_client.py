# coding=utf-8
# !/usr/bin/env python

from socket import *
from time import sleep
import itertools
from random import choice
from copy import copy

address = ('10.21.17.207', 16000)

s = socket(AF_INET, SOCK_DGRAM)

"""
北京市：117.136.0.0 - 117.136.0.255
广州市：58.66.72.0 - 58.66.72.255
长春市：58.155.144.0 - 58.155.147.255
大连市：58.154.144.0 - 58.154.159.255
西安市：58.206.96.0 - 58.206.96.255
武汉市：58.19.17.115 - 58.19.17.255
南宁市：58.67.159.0 - 58.67.159.255
温州市：60.12.58.0 - 60.12.58.255
银川市; 59.76.192.0 - 59.76.192.255
青岛市：58.56.128.0 - 58.56.128.255
拉萨市：124.31.0.0 - 124.31.0.255
上海市：59.155.185.0 - 59.155.185.255
重庆市：58.17.128.0 - 58.17.128.255
贵阳市：58.16.0.0 - 58.16.0.255

徐州市：202.195.64.0 - 202.195.64.255
苏州市：49.72.83.0 - 49.72.83.255
无锡市：58.193.120.0 - 58.193.120.255
扬州市：58.192.80.0 - 58.192.80.255

华盛顿：23.19.124.0 - 23.19.124.255
巴黎：80.11.106.0 - 80.11.106.255
"""

foreign_province = [
    '117.136.0.', '58.66.72.', '58.155.144.', '58.154.144.', '58.206.96.', '58.19.17.', '58.67.159.',
    '60.12.58.', '59.76.192.', '58.56.128.', '124.31.0.', '59.155.185.', '58.17.128.', '58.16.0.', '218.94.89.',
]

jiangsu_province = ['202.195.64.', '49.72.83.', '58.193.120.', '58.192.80.']
nanjing_head = '218.94.89.'
inner_head = '10.21.17.'

foreign_country = ['23.19.124.', '80.11.106.']

ip_tail = range(120, 150)
foreign_province = list(itertools.product(foreign_province, ip_tail))
foreign_province = map(lambda x: x[0] + str(x[1]), foreign_province)

jiangsu_province = list(itertools.product(jiangsu_province, ip_tail))
jiangsu_province = map(lambda x: x[0] + str(x[1]), jiangsu_province)

foreign_country = list(itertools.product(foreign_country, ip_tail))
foreign_country = map(lambda x: x[0] + str(x[1]), foreign_country)

nanjing = map(lambda i: nanjing_head + str(i), ip_tail)
inner = map(lambda i: inner_head + str(i), ip_tail)

malType = ['MALWARE', 'OTHERS', 'TROJAN', 'BACKDOOR', 'WORM', 'VIRUS', 'ADWARE', 'EXPLOIT', 'DOWNLOADER', 'DROPPER',
           'PSW', 'SPY']
worm = ['Net-WORM', 'Email-WORM', 'P2P-WORM']
troj = ['Win-TROJAN', 'DOS-TROJAN']
malName = ['Net-WORM', 'Email-WORM', 'P2P-WORM', 'Win-TROJAN', 'DOS-TROJAN']

pAttackPhase = ['Intelligence Gathering', 'Point of Entry', 'Command and Control Communication',
                'Lateral Movement', 'Asset and Data Discovery', 'Data Exfiltration']

sev = range(0, 11)
threatType_all = range(0, 7)
attack_threatType = [0, 1, 3]
deviceDirection = [0, 1]
cccaDetection = [0, 1]
data_dict = {}


def send_tda_data():
    base_data = "<191>LEEF:1.0|Asiainfo Security|Threat Discovery Appliance|3.71.1235|SECURITY_RISK_DETECTION|" \
                "devTimeFormat=MMM dd yyyy HH:mm:ss z\tptype=IDS\tdeviceMacAddress=00:0C:29:B6:8A:51\t" \
                "dvchost=localhost\tdeviceGUID=1ADB481E1C0C-41AB934E-3364-75F3-8F8F\t" \
                "devTime=May 29 2017 15:47:59 GMT+08:00\tprotoGroup=HTTP\tproto=HTTP\tvLANId=4095\tdstPort=80\t" \
                "dstMAC=00:0C:29:AF:2A:8F\tsrcPort=49595\tsrcMAC=30:37:A6:F6:5A:49\t" \
                "fileType=0\tfsize=0\truleId=1618\tmsg=CVE-2014-6271 - Shellshock HTTP Request\t" \
                "deviceRiskConfidenceLevel=1\tbotCommand=1070\tbotUrl= () { :;}; echo 'H0m3l4b1t: YES'\t" \
                "url=http://218.94.89.52:80/\trequestClientApplication=Python-urllib/2.4\tpComp=NCIE\triskType=1\t" \
                "srcZone=0\tdstGroup=10.0.0.0-10.255.255.255\tdstZone=1\tdetectionType=1\tact=not blocked\t" \
                "fileHash=0000000000000000000000000000000000000000\thostName=admin\tcnt=2\taggregatedCnt=1\t" \
                "evtCat=Exploit\tevtSubCat=Command Injection" \
                "\tdvc=%(dst)s\tdhost=%(dst)s\tdst=%(dst)s\tinterestedIp=%(dst)s" \
                "\tsrc=%(src)s\tshost=%(src)s\tpeerIp=%(src)s" \
                "\tmalType=%(malType)s" \
                "\tmalName=%(malName)s" \
                "\tsev=%(sev)s" \
                "\tdeviceDirection=%(deviceDirection)s" \
                "\tcccaDetection=%(cccaDetection)s" \
                "\tthreatType=%(threatType)s" \
                "\tpAttackPhase=%(pAttackPhase)s"

    for _ in range(100):
        base_dict = {'dst': choice(nanjing), 'src': choice(foreign_province), 'malType': choice(malType),
                     'malName': choice(malName), 'sev': choice(sev), 'deviceDirection': choice(deviceDirection),
                     'cccaDetection': choice(cccaDetection), 'threatType': choice(threatType_all),
                     'pAttackPhase': choice(pAttackPhase)}
        sleep(0.5)
        # 1、APT攻击
        # 1-1、阶段分布：造一些不同的pattackPhase_s
        data_dict = copy(base_dict)
        data_dict['deviceDirection'] = 0
        data_dict['dst'] = choice(inner)
        data_dict['src'] = choice(foreign_province)
        data_dict['pAttackPhase'] = pAttackPhase[0]
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        data_dict['src'] = choice(foreign_province)
        data_dict['pAttackPhase'] = pAttackPhase[1]
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        data_dict['src'] = choice(foreign_province)
        data_dict['pAttackPhase'] = pAttackPhase[2]
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        data_dict['src'] = choice(inner)
        data_dict['pAttackPhase'] = pAttackPhase[3]
        data_dict['deviceDirection'] = 1
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        data_dict['src'] = choice(foreign_province)
        data_dict['pAttackPhase'] = pAttackPhase[4]
        data_dict['deviceDirection'] = 0
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 1-2、类型分析：造一些不同的malwareType_s
        data_dict = copy(base_dict)
        data_dict['malType'] = choice(malType)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 2、网络攻击态势
        # 2-1、中国地图：目的地址是南京，并且threatType_s=0|1|3
        data_dict = copy(base_dict)
        data_dict['dst'] = choice(nanjing)
        data_dict['threatType'] = choice(attack_threatType)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 2-2、威胁类型：造一些不同的attackType_s
        data_dict = copy(base_dict)
        data_dict['threatType'] = choice(threatType_all)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 2-3、风险等级：造一些不同的severityLevel_s，并且threatType_s=0|1|3
        data_dict = copy(base_dict)
        data_dict['sev'] = choice(sev)
        data_dict['threatType'] = choice(attack_threatType)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 2-4、源地址TOP10：数据要求：多造些外省的srcIp_s，并且threatType_s=0|1|3
        data_dict = copy(base_dict)
        data_dict['src'] = choice(foreign_province)
        data_dict['threatType'] = choice(attack_threatType)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 2-5、目的地址TOP10：数据要求：多造些不同地市的dstIp_s，并且threatType_s=0|1|3
        data_dict = copy(base_dict)
        data_dict['dst'] = choice(foreign_province + foreign_country)
        data_dict['threatType'] = choice(attack_threatType)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 3、僵尸网络    江苏省地图(数据要求： 源地址是 外省地市；目的地址是江苏省各个地市)
        data_dict = copy(base_dict)
        data_dict['dst'] = choice(jiangsu_province)
        data_dict['src'] = choice(foreign_province)
        data_dict['cccaDetection'] = 1
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 4、蠕虫检测    蠕虫目的主机地址Top10;（数据要求: malwareName_s=*WORM* 并且目的地址是本省的各个地市）
        data_dict = copy(base_dict)
        data_dict['dst'] = choice(jiangsu_province)
        data_dict['malName'] = choice(worm)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)

        # 5、木马检测    木马目的主机地址Top10;（数据要求: malwareName_s=*TROJ* 并且目的地址是本省的各个地市）
        data_dict = copy(base_dict)
        data_dict['dst'] = choice(jiangsu_province)
        data_dict['malName'] = choice(troj)
        data_dict['pAttackPhase'] = choice(pAttackPhase)
        tda_data = base_data % data_dict
        s.sendto(tda_data, address)


send_tda_data()
