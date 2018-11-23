# coding=utf-8
import os
import time
import shutil
from ConfigParser import RawConfigParser
from multiprocessing import Process

import pymysql
from flask import Flask, request, json

from ml_train import ml_train
from ml_predict import ml_predict

app = Flask(__name__)


def train(**kwargs):
    time.sleep(1)
    # raise Exception


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def insert_data(table, kwargs):
    conn = pymysql.connect(**options)
    cur = conn.cursor()
    columns = ','.join(kwargs.keys())
    value = kwargs.values()
    sql = 'insert into {}({}) values({})'.format(table, columns, ','.join(['%s'] * len(value)))
    cur.execute(sql, value)
    conn.commit()

    cur.close()
    conn.close()


@app.route('/train', methods=['POST'])
def api_post_train():
    # curl -H "Content-type: application/json" -X POST http://10.21.37.198:7777/train -d
    # '{ "data_input_path":"/home/ml/caijr/ml-engine/data/iris.csv", "features":"label,a,b,c,d", "model_id":"101",
    # "train_nickname":"train_model_test","penalty":"l2", "solver":"saga", "multi_class":"multinomial","train_id":"112"}'

    assert request.headers['Content-Type'] == 'application/json', 'Post data must be json format!'
    data_json = request.json

    kwargs = {}
    fixed_keys = ['train_id', 'data_input_path', 'features', 'model_id', 'train_nickname']
    for _key in fixed_keys:
        if not data_json.has_key(_key):
            return '404, not exists key: ' + _key
        kwargs[_key] = data_json.pop(_key)
    kwargs['model_parameters'] = json.dumps(data_json)
    print kwargs
    # p = Process(target=ml_train, kwargs=kwargs)
    # p.daemon = True
    # p.start()
    insert_data('train_instances', kwargs)
    try:
        ml_train(**kwargs)
    except Exception as e:
        print e
        return '404'
    return '200'


@app.route('/offline_predict', methods=['POST'])
def api_post_test():
    # curl -H "Content-type: application/json" -X POST http://10.21.37.198:7777/predict -d
    # '{ "data_input_path":"/home/ml/caijr/ml-engine/data/iris.csv", "features":"label,a,b,c,d", "train_id":"1"}'
    assert request.headers['Content-Type'] == 'application/json', 'Post data must be json format!'
    data_json = request.json

    fixed_keys = ['train_id', 'data_input_path', 'features']
    for _key in fixed_keys:
        if not data_json.has_key(_key):
            return '404, not exists key: ' + _key

    features = data_json['features'].strip().split(',')
    with open(data_json['data_input_path']) as f:
        column_str = f.readline().strip()

    for f in features:
        if column_str.find(f) == -1:
            return '404, features of train and test do not match.'

    kwargs = dict()
    kwargs['train_id'] = data_json['train_id']
    kwargs['data_input_path'] = data_json['data_input_path']

    print kwargs
    # p = Process(target=ml_predict, kwargs=kwargs)
    # p.daemon = True
    # p.start()
    try:
        ml_predict(**kwargs)
    except Exception as e:
        print e
        return '404'
    return '200'


@app.route('/online_predict', methods=['POST'])
def api_post_test():
    # curl -H "Content-type: application/json" -X POST http://10.21.37.198:7777/predict -d
    # '{ "data_input_path":"/home/ml/caijr/ml-engine/data/iris.csv", "features":"label,a,b,c,d", "train_id":"1"}'
    assert request.headers['Content-Type'] == 'application/json', 'Post data must be json format!'
    data_json = request.json

    fixed_keys = ['train_id', 'address', 'topic']
    for _key in fixed_keys:
        if not data_json.has_key(_key):
            return '404, not exists key: ' + _key

    kwargs = dict()
    for _key in fixed_keys:
        kwargs[_key] = data_json[_key]

    print kwargs

    try:
        ml_online_predict(**kwargs)
    except Exception as e:
        print e
        return '404'
    return '200'


@app.route('/deploy', methods=['POST'])
def api_post_test():
    # curl -H "Content-type: application/json" -X POST http://10.21.37.198:7777/predict -d
    # '{ "data_input_path":"/home/ml/caijr/ml-engine/data/iris.csv", "features":"label,a,b,c,d", "train_id":"1"}'
    assert request.headers['Content-Type'] == 'application/json', 'Post data must be json format!'
    data_json = request.json

    fixed_keys = ['train_id', 'src_file', 'dst_file']
    for _key in fixed_keys:
        if not data_json.has_key(_key):
            return '404, not exists key: ' + _key

    try:
        shutil.copy(data_json['src_file'], data_json['dst_file'])

        conn = pymysql.connect(**options)
        cur = conn.cursor()
        table = 'train_instances'
        sql = "update {} set training_status=3 where train_id='{}'".format(table, data_json['train_id'])
        cur.execute(sql)
        conn.commit()

        cur.close()
        conn.close()
    except Exception as e:
        print e
        return '404'
    return '200'


if __name__ == '__main__':
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    app.run(host='0.0.0.0', port=7777, debug=True, use_evalex=False)
