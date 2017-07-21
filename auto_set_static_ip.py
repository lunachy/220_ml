# coding=utf-8

import os
from time import sleep

cmd = 'netsh interface ip set address Ethernet dhcp'

for i in range(210, 231):
    adapter = 'Ethernet'
    cmd = 'netsh interface ip set address {} static 10.21.37.{} 255.255.255.0 10.21.37.1 1'.format(adapter, str(i))
    print cmd
    os.system(cmd)

    os.system('netsh interface set interface {} disabled'.format(adapter))
    os.system('netsh interface set interface {} enabled'.format(adapter))

    sleep(5)
    ret = os.system('ping -n 4 www.baidu.com')
    if ret == 0:
        print("set ip address: 10.21.37.%s" % (str(i)))
        break
