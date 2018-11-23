from __future__ import print_function
import argparse
import json
import os
import pymysql
import ConfigParser
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans,DBSCAN,SpectralClustering
from sklearn.metrics import homogeneity_score,silhouette_score
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


models = {
           "301": KMeans,
           "302": DBSCAN,
           "303": SpectralClustering}
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
        cols = [col for col in data.columns if 'label' not in col]
        data = data[cols]
    else:
        # use_cols = map(lambda x: int(x), features.split(','))
        use_cols = map(lambda x: x, features.split(','))
        if 'label' in use_cols or 'Label' in use_cols:
            use_cols = use_cols[1:]
        data = pd.read_csv(data_input_path, usecols=use_cols)
    X = convert_categorical_by_dv(data,output_path)
    print(X.shape)
    return X


# def do_metrics_cluster(labels_true, labels_pred,X):
#     if labels_true.any():
#         print('homogeneity_score:',homogeneity_score(labels_true, labels_pred))
#     else:
#         print('silhouette_score:',silhouette_score(X, labels_pred, metric='euclidean'))

def do_metrics_cluster(labels_pred, X):
    evaluation = dict()
    # print('silhouette_score:', silhouette_score(X, labels_pred, metric='euclidean'))
    evaluation["silhouette_score"] = round(silhouette_score(X, labels_pred, metric='euclidean'),2)
    return evaluation


def do_plot_on_reduced_data_kmeans(data, clf, output_path):
    reduced_data = PCA(n_components=2).fit_transform(data)
    clf.fit(reduced_data)

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # plot the centroids as a white X
    centroids = clf.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on data (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(root_dir, output_path, 'cluster_plot.png'))
    plt.close()


def do_plot_on_reduced_data_others(data, clf, output_path):
    reduced_data = PCA(n_components=2).fit_transform(data)
    reduced_labels = clf.fit_predict(reduced_data)
    n_classes = set(reduced_labels)
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])

    plt.figure(1)
    plt.clf()
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    for i in n_classes:
        sub_reduced_data = reduced_data[reduced_labels==i]
        plt.scatter(sub_reduced_data[:, 0], sub_reduced_data[:, 1], s=10, color=next(prop_iter)['color'])
    plt.title('Clustering on PCA-reduced data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(root_dir, output_path, 'cluster_plot.png'))
    plt.close()


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


def ml_cluster(data_input_path, features, model_parameters, model_id, train_id):
    output_path = os.path.join(root_dir, 'models', train_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## read data
    X = read_data(data_input_path, features, output_path)

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

    # clf.fit(X)
    # labels_pred = clf.labels_
    labels_pred = clf.fit_predict(X)
    # labels_true = None
    evaluation = do_metrics_cluster(labels_pred, X)
    model_output_path = os.path.join(root_dir, output_path, "%s.pkl" % model_id)
    joblib.dump(clf, model_output_path)

    if model_id in ['301']:
        do_plot_on_reduced_data_kmeans(X, clf, output_path)
    elif model_id in ['302', '303']:
        do_plot_on_reduced_data_others(X, clf, output_path)

    training_status = 1
    mysql_output = [model_output_path, json.dumps(evaluation), training_status, train_id]

    # write to mysql
    _options = read_conf('settings.conf')
    mysql_db = pymysql.connect(**_options)
    update_mysql(mysql_db, mysql_output)

    print('finished')







