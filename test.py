#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import pymysql
from flask import Flask, request, jsonify, make_response, abort, json, Response
from ConfigParser import RawConfigParser

app = Flask(__name__)


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


class DB(object):
    def __init__(self):
        self.conn = pymysql.connect(**options)
        self.cur = self.conn.cursor()
        # self.num = self.cos.execute()

    def __enter__(self):
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()
        self.cur.close()


def json_error(status_code, message):
    """Return a JSON object with a HTTP error code."""
    r = jsonify(message=message)
    r.status_code = status_code
    return r


@app.route("/health")
def health():
    result = {'status': 'UP'}
    return Response(json.dumps(result), mimetype='application/json')


@app.route("/getUser")
def getUser():
    result = {'username': 'python', 'password': 'python'}
    return Response(json.dumps(result), mimetype='application/json')


@app.route("/file/<md5>")
@app.route("/url/<url>")
@app.route("/domain/<domain>")
@app.route("/ip/<ip>")
def result_view(md5=None, url=None, domain=None, ip=None):
    sql_base = "select signature, category_vt from {} where {}='{}'"
    response = {}
    if md5:
        response['md5'] = md5
        sql = sql_base.format('black_file', 'md5', md5)
    elif url:
        response['url'] = url
        sql = sql_base.format('black_url', 'url', url)
    elif domain:
        response['domain'] = domain
        sql = sql_base.format('black_domain', 'domain', domain)
    elif ip:
        response['ip'] = ip
        sql = sql_base.format('black_ip', 'ip', ip)
    else:
        return json_error(400, "Invalid lookup term")

    with DB() as cur:
        cur.execute(sql)
        result = cur.fetchone()

    if not result:
        return json_error(404, "sample not found")

    response['signature'] = result[0]
    response['category_vt'] = result[1]
    return jsonify(response)


if __name__ == '__main__':
    options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    app.run(host='0.0.0.0', port=7777, debug=True, use_evalex=False)
