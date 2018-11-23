# -*- coding: utf-8 -*-
import os
import datetime
from ftplib import FTP
import ConfigParser
import base64
import MySQLdb
def ftp_conf():
    config_parser = ConfigParser.ConfigParser()
    config_parser.read(os.path.join('update_config.conf'))
    ftp_info={}
    ftp_info['host'] = config_parser.get("ftp_server", "host")
    ftp_info['port']=int(config_parser.get("ftp_server", "port"))
    ftp_info['user'] = config_parser.get("ftp_server", "user")
    ftp_info['passwd']= base64.b64decode(config_parser.get("ftp_server", "pwd"))
    return ftp_info
def mysql_conf():
    config_parser = ConfigParser.ConfigParser()
    config_parser.read(os.path.join('update_config.conf'))
    mysql_info={}
    mysql_info['host'] = config_parser.get("mysq_server", "host")
    mysql_info['port']=int(config_parser.get("mysql_server", "port"))
    mysql_info['user'] = config_parser.get("mysql_server", "user")
    mysql_info['passwd']= config_parser.get("mysq_server", "passwd")
    mysql_info['dbname']=config_parser.get("mysq_server", "dbname")
    return mysql_info
def ftp_download():
    local_folder=os.path.join(os.getcwd(),'update_file')
    if os.path.exists(local_folder):
        pass
    else:
        os.mkdir(os.path.join(os.getcwd(),'update_file'))
    ftp_info=ftp_conf()
    ftp = FTP()
    ftp.set_debuglevel(2)  # 打开调试级别2，显示详细信息
    ftp.connect(ftp_info['host'], ftp_info['port'])  # 连接
    ftp.login(ftp_info['user'], ftp_info['passwd'])  # 登录，如果匿名登录则用空串代替即可
    table_name = ['black_ip', 'black_domain', 'black_url', 'black_file']
    today_date = str(datetime.datetime.now().strftime('%Y-%m-%d'))
    for one in table_name:
        file_name = today_date + one + '.sql'
        remotepath = "/data/cti/%s"%(file_name)
        file_local =os.path.join(local_folder,file_name)
        bufsize = 1024  # 设置缓冲器大小
        fp = open(file_local, 'wb')
        ftp.retrbinary('RETR %s' % remotepath, fp.write, bufsize)
        fp.close()
        #run_sql(remotepath)

def run_sql(path):
    mysql_info=mysql_conf()
    try:
        os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.utf8'
        db = MySQLdb.connect(host=mysql_info['host'] , port=int(mysql_info['port']) , user=mysql_info['user'] , passwd=mysql_info['passwd'], db=mysql_info['dbname'],charset="utf8")
        c = db.cursor()
        ##读取SQL文件,获得sql语句的list
        with open(path, 'r+') as f:
            sql_list = f.read().split(';')[:-1]  # sql文件最后一行加上;
            sql_list = [x.replace('\n', ' ') if '\n' in x else x for x in sql_list]  # 将每段sql里的换行符改成空格
        ##执行sql语句，使用循环执行sql语句
        for sql_item in sql_list:
            # print (sql_item)
            c.execute(sql_item)
    except Exception as e:
        print e
    finally:
        c.close()
        db.commit()
        db.close()


if __name__=="__main__":
    ftp_download()
    print 'finish'