#!/usr/bin/python
# coding=utf-8
import sys
import requests

reload(sys)
sys.setdefaultencoding("utf-8")

r = requests.get('http://10.21.37.198:8081/zhx_data.py')
with open('text.txt', 'w') as f:
    f.write(r.text)
