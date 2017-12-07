#!/usr/bin/python
# coding=utf-8
import os
import sys
import time
import json
import signal
import MySQLdb
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
import logging.handlers
import functools
from collections import Counter
import argparse
from sklearn.ensemble import RandomForestClassifier

root_dir = "/data/root/pe_classify/"
log_dir = "/data/log/"
report_dir = '/data/root/cuckoo/storage/analyses/'
signature_dir = '/data/root/signatures/'
train_csv = os.path.join(root_dir, '2017game_train.csv')
test_csv = os.path.join(root_dir, '2017game_test.csv')
train_md5_sig_encoding_path = os.path.join(root_dir, 'train_md5_sig_encoding.npz')
test_md5_sig_encoding_path = os.path.join(root_dir, 'test_md5_sig_encoding.npz')
train_md5_api_encoding_path = os.path.join(root_dir, 'train_md5_api_encoding.npz')
test_md5_api_encoding_path = os.path.join(root_dir, 'test_md5_api_encoding.npz')
CPU_COUNT = cpu_count()
train_count = 47067
test_count = 9398

CPU_COUNT = 10

log = logging.getLogger()
formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

fh = logging.handlers.WatchedFileHandler(os.path.join(log_dir, os.path.splitext(__file__)[0] + '.log'))
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)


def log_decorate(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        try:
            return_value = func(*args, **kw)
        except Exception as e:
            return_value = []
            log.info('error msg: %s', e)
            # log.info('error msg: %s, args: %s, kw: %s' %(e, args, kw))
        end = time.time()
        log.info('Called func[%s], starts: %s, costs %.2f seconds.' %
                 (func.__name__, time.strftime('%H:%M:%S', time.localtime(start)), (end - start)))
        return return_value

    return wrapper


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pool_map(func, iter_obj):
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    sigs = pool.map(func, iter_obj)
    pool.close()
    pool.join()
    return sigs


@log_decorate
def get_total_sig():
    def get_sig_name(py):
        with open(py) as f:
            sig_line = filter(lambda line: line.startswith('    name = '), f.readlines())
            sig_name = map(lambda s: s.strip().split('"')[1], sig_line)
            return sig_name

    os.chdir(signature_dir)
    all_sig_names = map(get_sig_name, os.listdir(signature_dir))
    return sum(all_sig_names, [])


def judge_signame(line):
    _line = line.strip()
    return True if _line.startswith('"name"') and line.endswith(',') else False


def get_sig(i):
    report_path = os.path.join(report_dir, str(i), 'reports/report.json')
    sigs_set = set()
    if os.path.exists(report_path):
        with open(report_path) as f:
            f.seek(900)
            for line in f:
                if line.startswith('            "name":'):
                    line_strip = line.strip()
                    if not line_strip.endswith(','):
                        sig = line_strip.split('"')[3]
                        sigs_set.add(sig)
                if line.startswith('    "target": {'):
                    break

    return sigs_set


@log_decorate
def multi_get_sig(start_id, end_id):
    return pool_map(get_sig, range(start_id, end_id + 1))


def get_api(i):
    report_path = os.path.join(report_dir, str(i), 'reports/report.json')
    api_str = ''
    apistats_flag = False
    api_dict = Counter()
    if os.path.exists(report_path):
        with open(report_path) as f:
            f.seek(900)
            for line in f:
                if line.startswith('        "apistats":'):
                    api_str = '{'
                    apistats_flag = True
                    continue
                if line.startswith('        "processes":'):
                    api_str = api_str[:-2]
                    break
                if apistats_flag:
                    api_str += line[:-1]

    if apistats_flag:
        _api_dict = json.loads(api_str)
        api_dict = reduce(lambda c1, c2: c1 + c2, map(lambda x: Counter(x), _api_dict.values()))
    return api_dict


@log_decorate
def multi_get_api(start_id, end_id):
    return pool_map(get_api, range(start_id, end_id + 1))


def get_md5_from_select(target):
    return target[0].rsplit('/', 1)[-1]


@log_decorate
def multi_get_md5s(start_id, end_id):
    conn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='root123', db='sandbox')
    cur = conn.cursor()
    _targets = cur.execute('select target from tasks where id >= {} and id <= {}'.format(start_id, end_id))
    targets = cur.fetchmany(_targets)
    cur.close()
    conn.close()

    return pool_map(get_md5_from_select, targets)


def reduce_sigs(sigs_list_set):
    return reduce(lambda x, y: x | y, sigs_list_set)


def reduce_apis(apis_list_counter):
    return reduce(lambda x, y: x | y, map(lambda api_c: set(api_c.keys()), apis_list_counter))


@log_decorate
def multi_encoding_api(apis_list_counter, all_apis):
    def _encode(apis_counter):
        row = [0] * len(all_apis)
        for api in apis_counter:
            if api in all_apis:
                row[all_apis.index(api)] = apis_counter[api]
        return row

    return map(_encode, apis_list_counter)


@log_decorate
def multi_encoding_sig(sigs_list_set, all_sigs):
    def _encode(sigs_set):
        row = [0] * len(all_sigs)
        for sig in sigs_set:
            if sig in all_sigs:
                row[all_sigs.index(sig)] = 1
        return row

    return map(_encode, sigs_list_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="get api from report", action="store_true", required=False)
    parser.add_argument("--sig", help="get signature from report", action="store_true", required=False)
    parser.add_argument("--destroy", help="destroy vms", action="store_true", required=False)
    parser.add_argument("--list_snapshot", help="list snapshot", action="store_true", required=False)
    parser.add_argument("--rm_vm", help="remove vm", action="store_true", required=False)
    parser.add_argument("--test_vm", help="test vm", action="store_true", required=False)
    parser.add_argument("--kvm_conf", help="generate kvm.conf", action="store_true", required=False)
    parser.add_argument("--gen_rem", help="generate rem hostname ip mac", action="store_true", required=False)
    parser.add_argument("--domain", type=str, help="domain", action="store", required=False, default=None)
    parser.add_argument("-p", "--parallel", type=int, help="number of parallel threads", action="store", required=False,
                        default=4)
    args = parser.parse_args()

    # get md5s from mysql
    train_md5s = multi_get_md5s(1, train_count)
    test_md5s = multi_get_md5s(train_count + 1, train_count + test_count)
    # all_sigs = list(set(get_total_sig()))

    if args.api:
        # get api from sandbox report
        train_apis = multi_get_api(1, train_count)
        all_apis = list(reduce_apis(train_apis))
        train_apis_encoding = multi_encoding_api(train_apis, all_apis)
        np.savez_compressed(train_md5_api_encoding_path, md5=train_md5s, api=train_apis_encoding, all_api=all_apis)
        log.info('length of train apis: %s', len(all_apis))

        test_apis = multi_get_api(train_count + 1, train_count + test_count)
        test_apis_encoding = multi_encoding_api(test_apis, all_apis)
        np.savez_compressed(test_md5_api_encoding_path, md5=test_md5s, api=test_apis_encoding)


    if args.sig:
        # get signature from sandbox report
        train_sigs = multi_get_sig(1, train_count)
        all_sigs = list(reduce_sigs(train_sigs))
        train_sigs_encoding = multi_encoding_sig(train_sigs, all_sigs)
        np.savez_compressed(train_md5_sig_encoding_path, md5=train_md5s, sig=train_sigs_encoding, all_sig=all_sigs)

        test_sigs = multi_get_sig(train_count + 1, train_count + test_count)
        test_sigs_encoding = multi_encoding_sig(test_sigs, all_sigs)
        np.savez_compressed(test_md5_sig_encoding_path, md5=test_md5s, sig=test_sigs_encoding)

    train_label = pd.read_csv(train_csv)
    train_md5_api_encoding = np.load(train_md5_api_encoding_path)
    train_api = pd.DataFrame(zip(train_md5_api_encoding['md5'], train_md5_api_encoding['api']), columns=['md5', 'api'])
    md5_label_api = pd.merge(train_label, train_api, 'outer', 'md5')
    clf = RandomForestClassifier(n_jobs=4)
    clf.fit(X_train, y_train)

    y_pred_all4 = clf.predict(X_test)
    print(np.mean(y_pred_all4 == y_test))
