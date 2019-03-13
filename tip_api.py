#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import pymysql
from flask import Flask, request, jsonify, make_response, abort, json, Response, abort
from ConfigParser import RawConfigParser
import re

app = Flask(__name__)

ipv4_reg = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
ipv6_reg = re.compile('^([\da-fA-F]{1,4}:){7}[\da-fA-F]{1,4}$')


def json_error(status_code, message):
    """Return a JSON object with a HTTP error code."""
    r = jsonify(message=message)
    r.status_code = status_code
    return r


def read_conf(conf_path):
    cfg = RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


# @app.before_request
# def check_authentication():
#     token = '123'
#     auth = request.headers.get("NS-NTIP-KEY")
#     if not auth:
#         return json_error(401, "No NTI key")
#     if auth != token:
#         return json_error(401, "Invalid NTI key")


@app.route("/getUser")
def getUser():
    result = {'username': 'python', 'password': 'python'}
    return Response(json.dumps(result), mimetype='application/json')


# curl 'https://127.0.0.1:7778/api/v4/objects/ip-details/?query=192.155.89.148&type=ip-basic'
#   --cacert ca/ca-cert.pem
@app.route("/api/v4/objects/ip-details1/")
def ip_result():
    response = {}

    # 查询条件中有其余key，则返回查询条件不支持
    for _key in request.args.keys():
        if _key not in ['query', 'type']:
            return json_error(400, "Unsupported condition")

    # 查询条件中query或者type有一个为空，则返回请求格式错误
    query = request.args.get('query')
    type = request.args.get('type')
    if not (query and type):
        return json_error(400, "Incorrect format")

    response['query'] = query
    response['type'] = type

    # 如果ip地址不是ipv4/ipv6，则返回参数错误
    if ipv4_reg.search(query):
        ip_type = 'ipv4-addr'
    elif ipv6_reg.search(query):
        ip_type = 'ipv6-addr'
    else:
        return json_error(400, "Format error")

    # 如果查询类型不是预定义中的一种，则返回参数错误
    types = ['ip-basic', 'asset-security', 'indicator']
    if type not in types:
        return json_error(400, "Format error")

    if type == types[0]:  # IP基础情报
        keys = 'object,locations,whoises,ases,domain_count,domains,device_count,devices,' \
               'os_count,oses,service_count,services,accesses,industries,businesses,tags'
        response['object'] = ip_type
    elif type == types[1]:  # IP资产威胁情报
        keys = 'type,id,created_by,modified,tags,object,status,histories,related_count,related'
        # response['object'] = ip_type
    elif type == types[2]:  # IP威胁指示器情报
        keys = 'type,id,created_by,created,modified,revoked,confidence,categories,Name,' \
               'description,Operator,observables,Dir,pattern,valid_from,valid_until,' \
               'kill_chain_phases,threat_types,threat_level,credit_level,compromised,' \
               'act_types,Country,Region,City,related_count,Related'
    return jsonify(response)


# curl 'https://127.0.0.1:7778/api/v4/objects/ip-details/?query=192.155.89.148&type=ip-basic'
#   --cacert ca/ca-cert.pem
@app.route("/api/v4/objects/")
@app.route("/api/v4/objects/<threat_type>-details/")
def idu_result(threat_type=None):
    threat_types = ['ip', 'domain', 'url', 'sample']
    query = request.args.get('query')
    type = request.args.get('type')
    # response = dict(zip(['threat_type', 'query', 'type'], [threat_type, query, type]))
    # return jsonify(response)
    response = {}

    # 查询条件中有其余key，则返回查询条件不支持
    for _key in request.args.keys():
        if _key not in ['query', 'type']:
            return json_error(400, "Unsupported condition")

    # 查询条件中query或者type有一个为空，则返回请求格式错误
    query = request.args.get('query')
    type = request.args.get('type')
    if not (query and type):
        return json_error(400, "Incorrect format")

    response['query'] = query
    response['type'] = type

    if threat_type:
        if threat_type == threat_types[0]:  # ip
            # 如果ip地址不是ipv4/ipv6，则返回参数错误
            if ipv4_reg.search(query):
                ip_type = 'ipv4-addr'
            elif ipv6_reg.search(query):
                ip_type = 'ipv6-addr'
            else:
                return json_error(400, "Format error")

            # 如果查询类型不是预定义中的一种，则返回参数错误
            types = ['ip-basic', 'asset-security', 'indicator']
            if type not in types:
                return json_error(400, "Format error")

            if type == types[0]:  # IP基础情报
                keys = 'object,locations,whoises,ases,domain_count,domains,device_count,devices,' \
                       'os_count,oses,service_count,services,accesses,industries,businesses,tags'
                response['object'] = ip_type
            elif type == types[1]:  # IP资产威胁情报
                keys = 'type,id,created_by,modified,tags,object,status,histories,related_count,related'
                # response['object'] = ip_type
            elif type == types[2]:  # IP威胁指示器情报
                keys = 'type,id,created_by,created,modified,revoked,confidence,categories,Name,' \
                       'description,Operator,observables,Dir,pattern,valid_from,valid_until,' \
                       'kill_chain_phases,threat_types,threat_level,credit_level,compromised,' \
                       'act_types,Country,Region,City,related_count,Related'
        elif threat_type == threat_types[1]:  # domain
            pass
        elif threat_type == threat_types[2]:  # url
            pass
        elif threat_type == threat_types[3]:  # sample
            pass
        else:
            pass
    else:
        pass
    return jsonify(response)


# curl 'https://127.0.0.1:7778/api/v4/objects/domain-details/?query=cdn2.downloadsoup.com&type=domain-basic'
#   --cacert ca/ca-cert.pem
@app.route("/api/v4/objects/domain-details/")
def domain_result():
    response = {}


if __name__ == '__main__':
    # options = read_conf(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.conf'))
    app.run(host='0.0.0.0', port=7778, debug=True, use_evalex=False,
            ssl_context=("/data/chy/server/server-cert.pem", "/data/chy/server/server-key.pem"))
