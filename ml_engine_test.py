# coding=utf-8
import os
import sys
import json
import time
from itertools import product
from collections import OrderedDict
from uuid import uuid1
import logging.handlers

log = logging.getLogger('brute_force')


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


model_paras = OrderedDict([
    ('101', [
        {'penalty': ['l1', 'l2']},
        {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
        {'multi_class': ['ovr', 'multinomial']},
        {'class_weight': ['balanced']},
    ]),
    ('102', [
        # {'prior': [None]}
    ]),
    ('103', [
        {'criterion': ['gini', 'entropy']},
        {'splitter': ['best', 'random']},
        {'max_depth': [5, 10, 100]},
        {'min_samples_split': [2, 10, 20]},
        # min_samples_split must be an integer greater than 1 or a float in (0.0, 1.0];
        {'min_samples_leaf': [1, 5, 10]},
        {'class_weight': ['balanced']},
    ]),
    ('104', [
        {'n_estimators': [5, 10, 50]},
        {'criterion': ['gini', 'entropy']},
        {'class_weight': ['balanced']},
        {'max_features': ['auto', 'sqrt', 'log2']},
        {'max_depth': [5, 10, 100]},
        {'min_samples_split': [2, 10, 20]},
        {'min_samples_leaf': [1, 5, 10]},
    ]),
    ('105', [
        {'C': [0.3, 1.0, 3.0]},
        {'kernel': ['linear', 'rbf', 'poly', 'sigmoid']},
        {'gamma': ['auto', 0.03, 0.1]},
        {'coef0': [0.0, 0.03, 0.1]},
        {'degree': [1, 3, 5]},
        {'class_weight': ['balanced']},
    ]),
    ('106', [
        {'hidden_layer_sizes': [(100,), (10, 10)]},
        {'solver': ['lbfgs', 'sgd', 'adam']},
        {'alpha': [0.00001, 0.0001, 0.001]},
        {'batch_size': ['auto', 100, 200, 500]},
        {'learning_rate_init': [0.0001, 0.001, 0.01]},
    ]),
    ('201', [
        {'fit_intercept': [True, False]},
        {'normalize': [True, False]},
    ]),
    ('202', [
        {'criterion': ['mse', 'friedman_mse', 'mae']},
        {'splitter': ['best', 'random']},
        {'max_depth': [5, 10, 100]},
        {'min_samples_split': [2, 10, 20]},
        {'min_samples_leaf': [1, 5, 10]},

    ]),
    ('203', [
        {'n_estimators': [5, 10, 50]},
        {'criterion': ['mse', 'friedman_mse', 'mae']},
        {'max_features': ['auto', 'sqrt', 'log2']},
        {'max_depth': [5, 10, 100]},
        {'min_samples_split': [2, 10, 20]},
        {'min_samples_leaf': [1, 5, 10]},
    ]),
    ('204', [
        {'C': [0.3, 1.0, 3.0]},
        {'epsilon': [0.01, 0.1, 1.0]},
        {'kernel': ['linear', 'rbf', 'poly', 'sigmoid']},
        {'gamma': ['auto', 0.03, 0.1]},
        {'coef0': [0.0, 0.03, 0.1]},
        {'degree': [1, 3, 5]},
    ]),
    ('205', [
        {'hidden_layer_sizes': [(100,), (10, 10)]},
        {'solver': ['lbfgs', 'sgd', 'adam']},
        {'alpha': [0.00001, 0.0001, 0.001]},
        {'batch_size': ['auto', 100, 200, 400]},
        {'learning_rate_init': [0.0001, 0.001, 0.01]},
    ]),
    ('301', [
        {'n_clusters': [5, 8, 15]},
    ]),
    ('302', [
        {'eps': [0.1, 0.5, 1]},
        {'min_samples': [5, 10, 20]},
    ]),
    ('303', [
        {'n_clusters': [5, 8, 15]},
        {'affinity': ['nearest_neighbors', 'rbf']},
        {'gamma': [0.3, 1.0, 3.0]},
        {'n_neighbors': [5, 10, 20]},
    ])
])

if __name__ == '__main__':
    init_logging('/root/chy/ml_engine_test.log')
    cmd_model = """
        curl -H "Content-type: application/json" -X POST http://10.21.37.198:7777/{} -d '{}'
        """
    train_path = '/root/chy/train_classification.csv'
    features = ''

    # train
    for model_id in model_paras:
        paras_list = model_paras[model_id]
        paras_keys = map(lambda x: x.keys()[0], paras_list)
        paras_values = map(lambda x: x.values()[0], paras_list)
        for _paras in product(*paras_values):
            _paras_d = dict(zip(paras_keys, _paras))
            if 'penalty' in _paras_d and 'solver' in _paras_d:
                if _paras_d['penalty'] == 'l1' and _paras_d['solver'] in ['newton-cg', 'lbfgs', 'sag']:
                    break
            if model_id.startswith('1'):
                train_path = '/root/chy/train_classification.csv'
                features = 'label,' + ','.join(map(str, range(0, 100)))  # 2838
            if model_id.startswith('2'):
                train_path = '/root/chy/train_regression.csv'
                features = 'label,h,i,j,k,l,m,n,o,p,q'
            if model_id.startswith('3'):
                train_path = '/root/chy/train_cluster.csv'
                features = ','.join(map(str, range(0, 100)))  # 2838
            _data = _paras_d.update({'data_input_path': train_path, 'features': features, 'model_id': model_id,
                                     'train_nickname': 'train_nickname', 'train_id': str(uuid1())})
            _paras_str = json.dumps(_paras_d)
            cmd = cmd_model.format('train', _paras_str)
            retcode = os.popen(cmd).readlines()[0]
            if retcode != '200':
                log.info(cmd)
            time.sleep(2)
            break
