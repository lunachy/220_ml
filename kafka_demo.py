#!/usr/bin/python
# coding=utf-8
import argparse
import time
from kafka import KafkaProducer, KafkaConsumer

kafka_addr = '10.21.37.198:9092'
topic = 'test1'
proc_file = '/data/chy/dns_test.txt'


def kafka_cmd():
    """
    启动zookeeper
    bin/zookeeper-server-start.sh config/zookeeper.properties &

    启动kafka
    bin/kafka-server-start.sh config/server.properties &

    停止kafka
    bin/kafka-server-stop.sh

    停止zookeeper
    bin/zookeeper-server-stop.sh

    新建topic
    bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test1

    列出topic
    bin/kafka-topics.sh --list --zookeeper localhost:2181
    """
    pass


def kafka_producer():
    producer = KafkaProducer(bootstrap_servers=kafka_addr)
    with open(proc_file) as f:
        for line in f:
            producer.send(topic, line)


def kafka_consumer():
    consumer = KafkaConsumer(topic, bootstrap_servers=kafka_addr, auto_offset_reset='earliest')
    for ct, msg in enumerate(consumer, 1):
        print(msg.value)
        time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--produce", help="produce kafka data", action="store_true", required=False)
    parser.add_argument("-c", "--consumer", help="consumer kafka data", action="store_true", required=False)
    args = parser.parse_args()
    if args.produce:
        kafka_producer()
    if args.consumer:
        kafka_consumer()
