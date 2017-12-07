#!/usr/bin/python
# coding=utf-8

import argparse
import os

root_path = '/home/ml/chy/'
log_file = os.path.join(root_path, 'monitor_process.log')
# proc_name="craw_cnnvd.py"  #进程名
# file_name="/home/ml/chy/restart_craw_cnnvd.py.log"
# pid=0
# ()
# {
#   num=`ps -ef | grep $proc_name | grep -v grep | wc -l`
#   return $num
# }
# proc_id()
# {
#     pid=`ps -ef | grep $proc_name | grep -v grep | awk '{print $2}'`
# }
# proc_num
# number=$?
# if [ $number -eq 0 ]                                    # 判断进程是否存在
# then
#     `python /home/ml/chy/craw_cnnvd.py -a`
#     proc_id                                         # 获取新进程号
#     echo ${pid}, `date` >> $file_name      # 将新进程号和重启时间记录
#
# fi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('process', help="process monitored")
    parser.add_argument("-i", "--increment", help="crawl increment urls", action="store_true", required=False)
    parser.add_argument("-f", "--fail", help="crawl failed urls", action="store_true", required=False)
    parser.add_argument("-p", "--proxy", help="crawl proxy address", action="store_true", required=False)
    args = parser.parse_args()

    proc_num = os.popen("ps -ef | grep {} | grep -v grep | grep -v {} | wc -l".format(args.process)).read()
    print proc_num