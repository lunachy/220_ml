# coding=utf-8
import os
import time
from flask import Flask, request, json
from multiprocessing import Process
import pymysql

app = Flask(__name__)


def train(**kwargs):
    time.sleep(1)
    print kwargs


def insert_data(table, kwargs):
    conn = pymysql.connect(host='10.21.37.198', port=3306, user='ml', passwd='123456', db='ML_ENGINE', charset="utf8")
    cur = conn.cursor()
    columns = ','.join(kwargs.keys())
    value = kwargs.values()
    sql = 'insert into {}({}) values({})'.format(table, columns, ','.join(['%s'] * len(value)))
    cur.execute(sql, value)
    conn.commit()

    cur.close()
    conn.close()


@app.route('/', methods=['POST'])
def api_post_train():
    # curl -H "Content-type: application/json" -X POST http://10.21.37.198:7777/ -d
    # '{"train_id":"123", "data_input_path":"/root/123.txt", "features":"1,2,3", "model_id":"abc123",
    # "model_output_path":"/root/out_model", "train_nickname":"train_model_test","C":"1","alpha":"0.1"}'
    assert request.headers['Content-Type'] == 'application/json', 'Post data must be json format!'
    data_json = request.json

    kwargs = {}
    fixed_keys = ['train_id', 'data_input_path', 'features', 'model_id', 'model_output_path', 'train_nickname']
    for _key in fixed_keys:
        assert data_json.has_key(_key), 'not exists key: ' + _key
        kwargs[_key] = data_json.pop(_key)
    kwargs['model_parameters'] = json.dumps(data_json)
    p = Process(target=train, kwargs=kwargs)
    # p.daemon = True
    p.start()
    insert_data('train_instances', kwargs)
    return json.dumps(kwargs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True, use_evalex=False)
