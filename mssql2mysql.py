#!/usr/bin/python
# coding=utf-8
__author = 'chy'
import os
import time
import MySQLdb

email_sql = 'dbo.st_cn_emails.Table.sql'
big_story_sql = 'dbo.st_cn_big_story.Table.sql'
urls_sql = 'dbo.st_cn_urls.Table.sql'
vulnerabilities_sql = 'dbo.st_cn_vulnerabilities.Table.sql'
files_sql = 'dbo.st_cn_files.Table.sql'
threats_sql = 'mt_cn_threats.sql'
date_format = '2009-10-27 22:00:00'
len_df = len(date_format)


def format_data(data):
    data = data.strip()
    _i = data.find('N\'')
    if _i != -1:
        data = data[_i + 2: -1]
    if data in ['NULL', '', 'none']:
        data = None
    if data:
        _i = data.find('AS DateTime')
        if _i != -1:
            data = data.split('.')[0]
    return data


def email(sql_file):
    with open(sql_file) as f:
        content = ''
        for line in f:
            line = line.strip()
            if line.startswith('INSERT'):
                if content.find('VALUES') != -1:
                    content = content[content.index('VALUES') + 8: -1]
                    data = []
                    while 1:
                        try:
                            _data, left = content.split(',', 1)
                            if _data.count('\'') % 2:
                                _data, left = content.split('\',', 1)
                                _data += '\''
                            data.append(_data)
                            content = left
                        except ValueError:
                            data.append(content)
                            break
                    data = map(format_data, data)
                    if len(data) > 32:
                        data = data[:8] + [' '.join(data[8:-23])] + data[-23:]

                    try:
                        cur.execute(
                            'insert into emails(ST_EMAIL_ID, SPAM_NAME, SPAM_CATEGORY, EMAIL_LANGUAGE, BIG_STORY, ATTACHMENT_INFO, ATTACHMENT_NAME, ATTACHMENT_TYPE, EMAIL_INFO, RELATIONS, RELATION_MATRIX, NOTEWORTHY_INFO, TAGS, SCREENSHOTS, TMASE_INFO, SPAM_BLOCKING_DATE, PREVENTION, GLOSSARY_REFERENCE, FREE_TEXT, GENERIC_TEXT, RSS, ACTIVE, ACTIVATION_DATE, PLATFORM, LAST_MODIFIED_DATE, DATE_PUBLISHED, RELATED_BLOG_ENTRIES, OTHER_RESOURCES, ORIGINAL_TITLE, AUTHOR, MODIFIER, EDITOR) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            data)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            print e
                            print data
                            break

                    time.sleep(0.01)
                content = line
            else:
                content += line


def big_story(sql_file):
    with open(sql_file) as f:
        content = ''
        for line in f:
            # time.sleep(1)
            line = line.replace('\x00', '').strip()
            # print repr(line), line
            if line.startswith('INSERT'):
                if content.find('VALUES') != -1:
                    content = content[content.index('VALUES') + 8: -1]
                    data = []
                    while 1:
                        try:
                            _data, left = content.split(',', 1)
                            if _data.count('\'') % 2:
                                _data, left = content.split('\',', 1)
                                _data += '\''
                            data.append(_data)
                            content = left
                        except ValueError:
                            data.append(content)
                            break
                    data = map(format_data, data)
                    # if len(data) > 32:
                    #     data = data[:8] + [' '.join(data[8:-23])] + data[-23:]

                    try:
                        cur.execute(
                            'insert into big_story values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            data)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            # (1366, "Incorrect string value: '\\xA0</fon...' for column 'BIG_STORY_BODY' at row 1")
                            print e
                            print len(data)
                            # print data
                            # break

                    time.sleep(0.01)
                content = line
            else:
                content += line


def urls(sql_file):
    with open(sql_file) as f:
        content = ''
        for line in f:
            # time.sleep(1)
            line = line.replace('\x00', '').strip()
            # print repr(line), line
            if line.startswith('INSERT'):
                if content.find('VALUES') != -1:
                    content = content[content.index('VALUES') + 8: -1]
                    data = []
                    while 1:
                        try:
                            _data, left = content.split(',', 1)
                            if _data.count('\'') % 2:
                                _data, left = content.split('\',', 1)
                                _data += '\''
                            data.append(_data)
                            content = left
                        except ValueError:
                            data.append(content)
                            break
                    data = map(format_data, data)
                    # if len(data) > 32:
                    #     data = data[:8] + [' '.join(data[8:-23])] + data[-23:]

                    try:
                        cur.execute(
                            'insert into urls values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            data)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            # (1366, "Incorrect string value: '\\xA0PE_WI...' for column 'URL_DESCRIPTION' at row 1")
                            print e
                            print len(data)
                            # print data
                            # break

                    time.sleep(0.01)
                content = line
            else:
                content += line


def vulnerabilities(sql_file):
    with open(sql_file) as f:
        content = ''
        for line in f:
            # time.sleep(1)
            line = line.replace('\x00', '').strip()
            # print repr(line), line
            if line.startswith('INSERT'):
                if content.find('VALUES') != -1:
                    content = content[content.index('VALUES') + 8: -1]
                    data = []
                    while 1:
                        try:
                            _data, left = content.split(',', 1)
                            if _data.count('\'') % 2:
                                _data, left = content.split('\',', 1)
                                _data += '\''
                            data.append(_data)
                            content = left
                        except ValueError:
                            data.append(content)
                            break
                    data = map(format_data, data)
                    # if len(data) > 32:
                    #     data = data[:8] + [' '.join(data[8:-23])] + data[-23:]

                    try:
                        cur.execute(
                            'insert into vulnerabilities values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            data)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            # (1366, "Incorrect string value: '\\xA0PE_WI...' for column 'URL_DESCRIPTION' at row 1")
                            print e
                            print len(data)
                            # print data
                            # break

                    time.sleep(0.01)
                content = line
            else:
                content += line


def files(sql_file):
    with open(sql_file) as f:
        content = ''
        for line in f:
            # time.sleep(1)
            line = line.replace('\x00', '').strip()
            # print repr(line), line
            if line.startswith('INSERT'):
                if content.find('VALUES') != -1:
                    content = content[content.index('VALUES') + 8: -1]
                    data = []
                    while 1:
                        try:
                            _data, left = content.split(',', 1)
                            if _data.count('\'') % 2:
                                _data, left = content.split('\',', 1)
                                _data += '\''
                            data.append(_data)
                            content = left
                        except ValueError:
                            data.append(content)
                            break
                    data = map(format_data, data)
                    # if len(data) > 32:
                    #     data = data[:8] + [' '.join(data[8:-23])] + data[-23:]

                    try:
                        cur.execute(
                            'insert into files values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            data)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            # (1366, "Incorrect string value: '\\xA0PE_WI...' for column 'URL_DESCRIPTION' at row 1")
                            print e
                            print len(data)
                            # print data
                            # break

                    time.sleep(0.01)
                content = line
            else:
                content += line


def threats(sql_file):
    with open(sql_file) as f:
        content = ''
        for line in f:
            # time.sleep(1)
            line = line.replace('\x00', '').strip()
            # print repr(line), line
            if line.startswith('INSERT'):
                if content.find('VALUES') != -1:
                    content = content[content.index('VALUES') + 8: -1]
                    data = []
                    while 1:
                        try:
                            _data, left = content.split(',', 1)
                            if _data.count('\'') % 2:
                                _data, left = content.split('\',', 1)
                                _data += '\''
                            data.append(_data)
                            content = left
                        except ValueError:
                            data.append(content)
                            break
                    data = map(format_data, data)
                    # if len(data) > 32:
                    #     data = data[:8] + [' '.join(data[8:-23])] + data[-23:]

                    try:
                        cur.execute(
                            'insert into threats values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            data)
                        conn.commit()
                    except Exception, e:
                        if e.args[0] == 1062:
                            pass
                        else:
                            # (1366, "Incorrect string value: '\\xA0PE_WI...' for column 'URL_DESCRIPTION' at row 1")
                            print e
                            print len(data)
                            print data
                            break

                    time.sleep(0.01)
                content = line
            else:
                content += line


if __name__ == '__main__':
    mssql_dir = '/data/samba/'
    os.chdir(mssql_dir)
    conn = MySQLdb.connect(host='10.21.37.198', port=3306, user='ml',
                           passwd='123456', db='trend_data', charset='utf8')
    cur = conn.cursor()
    # email(email_sql)
    # big_story(big_story_sql)
    # urls(urls_sql)
    # vulnerabilities(vulnerabilities_sql)
    files(files_sql)
    threats(threats_sql)
    cur.close()
    conn.close()
