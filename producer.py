#!/usr/bin/python
# coding=utf-8
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='10.15.42.23:9092')
for _ in range(10):
    producer.send('leontest', b'some_message_bytes')
