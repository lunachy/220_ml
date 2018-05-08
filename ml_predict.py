from __future__ import print_function
import argparse
import json
import os
import pymysql
import ConfigParser
import logging.handlers
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.ensemble as ek
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.cluster import KMeans,DBSCAN,SpectralClustering
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import homogeneity_score,silhouette_score
from sklearn.externals import joblib


root_dir = '/home/ml/caijr/cjr-engine'
log = logging.getLogger()
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

fh = logging.handlers.WatchedFileHandler(os.path.join(root_dir, os.path.splitext(__file__)[0] + '.log'))
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)


def read_conf(conf_path):
    cfg = ConfigParser.RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def get_info_mysql(db, train_id):
    cur = db.cursor()
    sql = "SELECT model_id, model_output_path, features FROM train_instances WHERE train_id={0}".format(train_id)
    cur.execute(sql)
    data = cur.fetchone()
    cur.close()
    db.close()
    return data


def convert_categorical_by_dv_predict(X_test,vec_path):
    vec_loaded = joblib.load(vec_path)
    X_test_dict = X_test.to_dict(orient='records')
    vec_X_test = vec_loaded.transform(X_test_dict).toarray()
    return vec_X_test


def read_data(data_input_path, features, vec_path):
    if not features:
        data = pd.read_csv(data_input_path, index_col=0)
    else:
        use_cols = map(lambda x: x, features.split(','))
        if 'label' in use_cols or 'Label' in use_cols:
            use_cols = use_cols[1:]
        data = pd.read_csv(data_input_path, usecols=use_cols)
    print(data.shape)
    X = convert_categorical_by_dv_predict(data, vec_path)
    # y = data.iloc[:, -1]
    print(X.shape)
    return X


def ml_predict(train_id, data_input_path):
    _options = read_conf('settings.conf')
    mysql_db = pymysql.connect(**_options)
    model_id, model_output_path, features = get_info_mysql(mysql_db, train_id)
    print(model_id, model_output_path, features)

    # new data
    vec_path = os.path.join(os.path.dirname(model_output_path), 'vec.pkl')
    print(vec_path)
    data = read_data(data_input_path, features, vec_path)  # three situations by model_id
    x_features = data

    clf_loaded = joblib.load(model_output_path)

    if model_id.startswith("2"):
        y_pred = clf_loaded.predict(x_features)
    elif model_id.startswith("1"):
        if model_id in ['102', '103', '104', '106']:
            y_pred = clf_loaded.predict(x_features)
        elif model_id in ['101', '105']:
            y_pred = clf_loaded.predict(x_features)
            # y_pred = np.argmax(Y_pred, axis=1)
    elif model_id.startswith("3"):
        if model_id == '301':
            y_pred = clf_loaded.predict(x_features)
        elif model_id in ['302', '303']:
            y_pred = clf_loaded.fit_predict(x_features)

    # json results
    results = dict()
    for i in range(y_pred.shape[0]):
        results[i] = y_pred[i]

    if y_pred.shape[0] <= 10:
        print(results)
    else:
        print({k: results[k] for k in range(0, 10)})

        # # csv results
        # y_pred = pd.DataFrame(y_pred, columns=['predicted'])
        # data_p = pd.concat([y_pred, x_features], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_id", help="data_input_path", type=str, required=True)
    parser.add_argument("--data_input_path", help="data_input_path", type=str, required=True)
    options = parser.parse_args()

    time_format = "%Y-%m-%d  %H:%M:%S"
    logging.info('start time: %s !', time.strftime(time_format, time.localtime()))
    print(options.train_id)
    ml_predict(options.train_id, options.data_input_path)
    logging.info('finish time: %s !', time.strftime(time_format, time.localtime()))



