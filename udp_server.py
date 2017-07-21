#!/usr/bin/env python

from socket import *

HOST = '10.21.37.18'
PORT = 15000

s = socket(AF_INET, SOCK_DGRAM)
s.bind((HOST, PORT))
print '...waiting for message..'
while True:
    data, address = s.recvfrom(1024)
    print data, address
s.close()
