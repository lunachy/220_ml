from __future__ import print_function
import argparse
import json
import os
import pymysql
import ConfigParser
import numpy as np
import pandas as pd
import logging.handlers
import time
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
import graphviz
from sklearn import tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


models = { "101":LogisticRegression,
           "102":GaussianNB,
           "103":DecisionTreeClassifier,
           "104":ek.RandomForestClassifier,
           "105":SVC,
           "106":MLPClassifier,
           "201":LinearRegression,
           "202":DecisionTreeRegressor,
           "203":ek.RandomForestRegressor,
           "204":SVR,
           "205":MLPRegressor,
           "301":KMeans,
           "302":DBSCAN,
           "303":SpectralClustering}
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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_id", help="data_input_path", type=str, required=True)
    parser.add_argument("--data_input_path", help="data_input_path", type=str, required=True)
    parser.add_argument("--features", help="", type=str, default='')
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
    print(X.shape)

    vec_X = convert_categorical_by_dv(X, output_path)
    print(vec_X.shape)

    classes = list(set(y))
    n_classes = len(classes)
    print(n_classes)
    if n_classes > 2:
        y = label_binarize(y, classes=classes) # pd.get_dummies(y) is not fit here    #
    print(y.shape)
    return vec_X, y, n_classes


def do_metrics_classification(y_test, y_pred, n_classes):
    evaluation = dict()
    print("metrics.accuracy_score")
    print(accuracy_score(y_test, y_pred))
    # print("metrics.precision_recall_fscore_support")
    # print(precision_recall_fscore_support(y_test, y_pred,average='micro'))
    precison,recall, _, _ = precision_recall_fscore_support(y_test, y_pred,average='micro')
    evaluation["precison"], evaluation['recall'] = round(precison,2), round(recall,2)
    # print("metrics.classification_report:")
    # print(classification_report(y_test, y_pred)) # equal to average='weighted'
    # if n_classes == 2:
    #     print("metrics.confusion_matrix:")
    #     print(confusion_matrix(y_test, y_pred))
    # else:
    #     pass
    # print precision_recall_fscore_support(y_test, y_pred,average='macro')
    return evaluation


def get_curve_data(Y_test, Y_score, n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if n_classes > 2:
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                                Y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], Y_score[:, i])

            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    Y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, Y_score,
                                                         average="micro")

    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #       .format(average_precision["micro"]))
    return precision, recall, average_precision, fpr, tpr, roc_auc


def do_pr_plot(precision, recall, average_precision, output_path):
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # plt.savefig(os.path.join(root_dir, output_path, 'pr_plot_2.png'))
    plt.savefig(os.path.join(output_path, 'pr_plot_2.png'))
    plt.close()


def do_roc_plot(fpr, tpr, roc_auc, output_path):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'roc_plot_2.png'))
    plt.close()


def do_pr_plot_multi(precision, recall, average_precision, n_classes, output_path):
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])

    plt.figure()

    lines = []
    labels = []
    l, = plt.plot(recall["micro"], precision["micro"], color=next(prop_iter)['color'], lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i in range(n_classes):
        l, = plt.plot(recall[i], precision[i], color=next(prop_iter)['color'], lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    # plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.legend(lines, labels, loc="lower right")
    plt.savefig(os.path.join(output_path, 'pr_plot_multi.png'))
    plt.close()


def do_roc_plot_multi(fpr, tpr, roc_auc, n_classes, output_path):
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color=next(prop_iter)['color'], linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=next(prop_iter)['color'], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'roc_plot_multi.png'))
    plt.close()


def do_decision_tree_plot(clf, output_path, max_depth=3):
    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=max_depth)
    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(output_path, 'dt_plot'))


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


def ml_classification(data_input_path, features, model_parameters, model_id, train_id):
    output_path = os.path.join(root_dir, 'models', train_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## read data
    X, Y, n_classes = read_data(data_input_path, features, output_path)

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

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
    if n_classes == 2:
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        if model_id in ['101', '105']:
            Y_score = clf.decision_function(X_test)
        else:
            Y_score = clf.predict_proba(X_test)[:,1]
        evaluation = do_metrics_classification(Y_test, Y_pred, n_classes)

        if model_id =='103':
            do_decision_tree_plot(clf, output_path, max_depth=3)

        precision, recall, average_precision, fpr, tpr, roc_auc = get_curve_data(Y_test, Y_score, n_classes)
        do_pr_plot(precision["micro"], recall["micro"], average_precision["micro"], output_path)
        do_roc_plot(fpr["micro"], tpr["micro"], roc_auc["micro"], output_path)


    elif n_classes > 2:
        if model_id in ['101', '105']:
            # clf = OneVsRestClassifier(clf)
            y_train = np.argmax(Y_train, axis=1)
            y_test = np.argmax(Y_test, axis=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            Y_score = clf.decision_function(X_test)
            evaluation = do_metrics_classification(y_test, y_pred, n_classes)
        elif model_id in ['102', '103', '104', '106']:
            y_train = np.argmax(Y_train, axis=1)
            y_test = np.argmax(Y_test, axis=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            Y_score = clf.predict_proba(X_test)
            evaluation = do_metrics_classification(y_test, y_pred, n_classes)

        if model_id =='103':
            do_decision_tree_plot(clf, output_path, max_depth=3)

        precision, recall, average_precision, fpr, tpr, roc_auc = get_curve_data(Y_test, Y_score, n_classes)
        do_pr_plot_multi(precision, recall, average_precision, n_classes, output_path)
        do_roc_plot_multi(fpr, tpr, roc_auc,  n_classes, output_path)

    model_output_path = os.path.join(output_path, "%s.pkl" % model_id)
    joblib.dump(clf, model_output_path)

    training_status = 1
    mysql_output = [model_output_path, json.dumps(evaluation), training_status, train_id]

    # write to mysql
    _options = read_conf('settings.conf')
    mysql_db = pymysql.connect(**_options)
    update_mysql(mysql_db, mysql_output)

    print('finished')


# if __name__=="__main__":
#     parser = build_parser()
#     options = parser.parse_args()
#     ml_classification(options.data_input_path, options.features, options.model_parameters, options.model_id, options.train_id)


