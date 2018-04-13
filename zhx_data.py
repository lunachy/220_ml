#!/usr/bin/python
# coding=utf-8

from random import choice, sample, random, randrange
from datetime import datetime, timedelta
import pymysql
import string

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
乌鲁木齐：124.117.118.0 -124.117.118.255
西宁：210.27.176.0 -210.27.176.255
香港:203.198.167.0 -203.198.167.255
澳门：202.175.34.0 -202.175.34.255
台北：140.112.0.0 -140.112.0.255

徐州市：202.195.64.0 - 202.195.64.255
苏州市：49.72.83.0 - 49.72.83.255
无锡市：58.193.120.0 - 58.193.120.255
扬州市：58.192.80.0 - 58.192.80.255

华盛顿：23.19.124.0 - 23.19.124.255
巴黎：80.11.106.0 - 80.11.106.255
"""

USER_AGENTS = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]

foreign_province = [
    '117.136.0.', '58.66.72.', '58.155.144.', '58.154.144.', '58.206.96.', '58.19.17.', '58.67.159.',
    '60.12.58.', '59.76.192.', '58.56.128.', '124.31.0.', '59.155.185.', '58.17.128.', '58.16.0.', '218.94.89.',
    '124.117.118.', '210.27.176.', '203.198.167.', '202.175.34.', '140.112.0.',
]

jiangsu_province = ['202.195.64.', '49.72.83.', '58.193.120.', '58.192.80.']
nanjing_head = '218.94.89.'
inner_head = '10.21.17.'

foreign_country = ['23.19.124.', '80.11.106.']

ips_header = foreign_province + jiangsu_province + foreign_country
ip_tail = range(2, 254)
ips = []
uas = []
for i in range(100):
    ips.append(choice(ips_header) + str(choice(ip_tail)))
    uas.append(choice(USER_AGENTS))

all_ip_uas = zip(ips, uas)
baopo_ip_uas = all_ip_uas
zhuangku_ip_uas = baopo_ip_uas[:2]

baopo_usernames = ['admin', 'user', 'root', 'abc', 'admin123', 'root123', 'administrator', 'admin1234']
zhuangku_usernames = ['chenel', 'chenfx', 'chenlei13', 'gaofh', 'helong', 'huangqj', 'huangwei5', 'lihl8', 'linqun',
                      'linqiang', 'linwt', 'liuww5', 'liyan7', 'liuyr', 'luyx', 'qiufj', 'qinmx', 'tangyd', 'weibd',
                      'wangjianjun', 'wuqf5', 'wangwei29', 'wuym2', 'yesheng', 'zhangcf', 'zhusy3', 'songff', 'xufeng7',
                      'huohm', 'shigz', 'fancheng', 'zhangshang', 'zhangyy19', 'guoqy3', 'yemh', 'chengyong', 'licj6',
                      'xuxin6', 'caijr', 'miaohq', 'liubin5', 'zhanggq', 'huangqing2', 'suyb', 'shenyf', 'zhaolz5',
                      'linst', 'chenyy', 'liubing5', 'chengbin', 'shenfei', 'weisw', 'linlt', 'shigp', 'zhengyan',
                      'macc', 'tangjun5', 'yehc', 'sunbin5', 'duanbb', 'liyz3', 'yezc', 'liubo5', 'xieyy', 'lumm',
                      'chengr', 'liusc5', 'tianran3', 'wangyf15', 'zhangsj9', 'zhangle', 'wangqh3', 'wupeng', 'luhan3',
                      'wangtr', 'herg', 'zhibb', 'lili']
# passwds = ['123456', 'abcd1234', '12345678', 'qwertyuiop', '1234567890']
passwds = [''.join(sample(string.digits + string.ascii_letters, choice(range(6, 11)))) for _ in range(1000)]
method = ['POST', 'GET']
ret_code = [200, 301, 404]
"""
10.21.37.224 - - [04/Apr/2018:10:22:54 +0800] "GET /login.do?passwd=passwdd&userid=username HTTP/1.1" 404 500 "-" "python-requests/2.13.0"
10.21.37.224 - - [04/Apr/2018:10:24:59 +0800] "POST /login.do?passwd=passwdd&userid=username HTTP/1.1" 200 500 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"
10.21.37.224 - - [04/Apr/2018:10:24:59 +0800] "GET /main.do HTTP/1.1" 301 500 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"
"""

webs = ['admin-header.do', 'media-upload.do', 'options-reading.do', 'upgrade.do', 'load-scripts.do', 'import.do',
        'link-manager.do', 'update.do', 'options-permalink.do', 'customize.do', 'widgets.do', 'nav-menus.do',
        'plugins.do', 'my-sites.do', 'edit-link-form.do', 'freedoms.do', 'update-core.do', 'privacy.do',
        'admin-ajax.do', 'profile.do', 'post-new.do', 'themes.do', 'link.do', 'custom-background.do', 'admin-footer.do',
        'edit-tags.do', 'ms-admin.do', 'edit-form-advanced.do', 'options-discussion.do', 'theme-install.do',
        'link-add.do', 'options.do', 'media.do', 'theme-editor.do', 'edit.do', 'edit-tag-form.do', 'link-parse-opml.do',
        'options-head.do', 'admin.do', 'ms-themes.do', 'users.do', 'ms-upgrade-network.do', 'tools.do',
        'async-upload.do', 'credits.do', 'media-new.do', 'menu.do', 'setup-config.do', 'edit-comments.do',
        'admin-post.do', 'about.do', 'options-writing.do', 'install.do', 'export.do', 'index.do', 'moderation.do',
        'revision.do', 'ms-options.do', 'upgrade-functions.do', 'install-helper.do', 'user-new.do', 'press-this.do',
        'edit-form-comment.do', 'menu-header.do', 'upload.do', 'network.do', 'ms-edit.do', 'load-styles.do',
        'ms-delete-site.do', 'term.do', 'options-general.do', 'plugin-editor.do', 'options-media.do', 'user-edit.do',
        'admin-functions.do', 'ms-sites.do', 'comment.do', 'post.do', 'custom-header.do', 'ms-users.do',
        'plugin-install.do']

msg1_model = '{} - - [{} +0800] "{} /login.do?passwd={}&userid={} HTTP/1.1" {} 500 "-" "{}"\n'
msg2_model = '{} - - [{} +0800] "{} /main.do HTTP/1.1" {} 500 "-" "{}"\n'
error_msg_model_1 = '{} - - [{} +0800] "{} /login.do?passwd={}&userid={} HTTP/1.1" {} 500 "-" "{}"\n'
error_msg_model_2 = '{} - - [{} +0800] "{} /{} HTTP/1.1" {} 500 "-" "{}"\n'


def pseudo_log(usernames, ip_uas, table_name, log_file):
    print datetime.now()
    # pre_time = '04/Apr/2018:01:01:01'
    # date_format = '%d/%b/%Y:%H:%M:%S'
    pre_time = '2018-04-04 01:01:20'
    date_format = '%Y-%m-%d %H:%M:%S'
    ptime = datetime.strptime(pre_time, date_format)
    conn = pymysql.connect(host='10.21.37.198', user='ml', passwd='123456', port=3306, db='SSA', charset='utf8')
    cur = conn.cursor()
    field_names = 'login_ip,login_time,user_name,login_status'
    sql = 'insert into {}({}) values(%s,%s,%s,%s)'
    delta_seconds = 0
    with open(log_file, 'w') as f:
        for _ in range(4000):
            # for _ in range(choice(range(1))):
            # if True:
            #     _ip, _ua = choice(ip_uas)
            #     # delta_seconds += round(random(round((_p * 1.0 / value[3] - 1) * 100, 2)
            #     delta_seconds += randrange(1, 5) / 10.0
            #     ftime = (ptime + timedelta(seconds=delta_seconds)).strftime(date_format)
            #     # correct msg
            #     _user, _pass = choice(usernames), choice(passwds)
            #     msg1 = msg1_model.format(_ip, ftime, 'POST', _pass, _user, 200, _ua)
            #     _value = [_ip, ftime, _user, 1]
            #     cur.execute(sql.format(table_name, field_names), _value)
            #     f.write(msg1)
            #     msg2 = msg2_model.format(_ip, ftime, 'GET', 301, _ua)
            #     f.write(msg2)
                # print msg1, msg2

            # error msg
            for _ in range(choice(range(10))):
                _ip, _ua = choice(ip_uas)
                delta_seconds += randrange(1, 5) / 10.0
                ftime = (ptime + timedelta(seconds=delta_seconds)).strftime(date_format)
                _eip = choice(ips_header) + str(choice(ip_tail))
                _user, _pass = choice(usernames), choice(passwds)
                error_msg = error_msg_model_1.format(_eip, ftime, 'POST', _pass, _user, choice(ret_code), _ua)
                _value = [_ip, ftime, _user, 0]
                cur.execute(sql.format(table_name, field_names), _value)
                f.write(error_msg)

            for _ in range(choice(range(4))):
                _ip, _ua = choice(all_ip_uas)
                delta_seconds += randrange(1, 5) / 10.0
                ftime = (ptime + timedelta(seconds=delta_seconds)).strftime(date_format)
                _eip = choice(ips_header) + str(choice(ip_tail))
                error_msg = error_msg_model_2.format(_eip, ftime, choice(method), choice(webs), choice(ret_code), _ua)
                f.write(error_msg)

    conn.commit()
    cur.close()
    conn.close()
    print datetime.now()


# 爆破，ip随机，用户名固定10个以内， 密码随机，错误为主
pseudo_log(baopo_usernames, baopo_ip_uas, 'baopo_log', 'baopo.log')
# 撞库，ip固定10个以内，用户名随机， 密码随机，错误为主
# pseudo_log(zhuangku_usernames, zhuangku_ip_uas, 'zhuangku_log', 'zhuangku.log')
