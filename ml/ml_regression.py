from __future__ import print_function
import argparse
import json
import os
import pymysql
import ConfigParser
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
import sklearn.ensemble as ek
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error,r2_score


models = {
           "201":LinearRegression,
           "202":DecisionTreeRegressor,
           "203":ek.RandomForestRegressor,
           "204":SVR,
           "205":MLPRegressor
           }
root_dir = '/home/ml/caijr/cjr-engine'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_id", help="data_input_path", type=str, required=True)
    parser.add_argument("--data_input_path", help="data_input_path", type=str, required=True)
    parser.add_argument("--features", help="", type=str, required=True)
    parser.add_argument("--model_id", help="", type=str, required=True)
    parser.add_argument("--model_parameters", default=None, help="", type=str)
    return parser


def convert_categorical_by_dv(X, output_path):
    vec = DictVectorizer()
    X_dict = X.to_dict(orient='records')
    vec_X = vec.fit_transform(X_dict).toarray() #type(vec_X):scipy.sparse.csr.csr_matrix
    # vec_X_columns = vec.get_feature_names()
    print(vec.get_feature_names())
    joblib.dump(vec, os.path.join(output_path, 'vec.pkl'))
    return vec_X


def read_data(data_input_path, features, output_path):
    if not features:
        data = pd.read_csv(data_input_path, index_col=0)
    else:
        # use_cols = map(lambda x: int(x), features.split(','))
        use_cols = map(lambda x: x, features.split(','))
        data = pd.read_csv(data_input_path, usecols=use_cols)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(X.shape, y.shape)

    vec_X = convert_categorical_by_dv(X, output_path)
    print(vec_X.shape)

    return vec_X, y


def do_metrics_regression(y_test, y_pred):
    evaluation = dict()
    # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
    # print('r2_score:', r2_score(y_test, y_pred))
    evaluation['RMSE'] = round(np.sqrt(mean_squared_error(y_test, y_pred)),2)
    return evaluation


def read_conf(conf_path):
    cfg = ConfigParser.RawConfigParser()
    cfg.read(conf_path)
    _keys = ['host', 'user', 'passwd', 'port', 'db', 'charset']
    _options = {}
    for _k in _keys:
        _options[_k] = cfg.get('mysql', _k).strip()
    _options['port'] = int(_options['port'])
    return _options


def update_mysql(db, output):
    cur = db.cursor()
    sql = "UPDATE train_instances SET model_output_path=%s,model_evaluation=%s,training_status=%s " \
          " WHERE train_id=%s "
    cur.execute(sql, output)
    db.commit()
    cur.close()
    db.close()


def ml_regression(data_input_path, features, model_parameters, model_id, train_id):
    output_path = os.path.join(root_dir, 'models', train_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## read data
    X, y = read_data(data_input_path, features, output_path)

    ## fetch model and model's parameters
    # kwargs = {"penalty": 'l2', "solver": "saga", "multi_class": 'multinomial', 'class_weight': None}
    if model_parameters:
        kwargs = json.loads(model_parameters)
        try:
            clf = models[model_id](**kwargs)
        except TypeError as e:
            print("model parameters do not fit model" + repr(e))
    else:
        clf = models[model_id]()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    evaluation = do_metrics_regression(y, y_pred)

    model_output_path = os.path.join(root_dir, output_path, "%s.pkl" % model_id)
    joblib.dump(clf, model_output_path)

    training_status = 1
    mysql_output = [model_output_path, json.dumps(evaluation), training_status, train_id]

    # write to mysql
    _options = read_conf('settings.conf')
    mysql_db = pymysql.connect(**_options)
    update_mysql(mysql_db, mysql_output)

    print('finished')
