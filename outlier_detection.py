# coding=utf-8
import numpy as np
from sklearn import svm
import pandas as pd

def oneclasssvm():
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data 生成训练数据
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    print(X_train.shape)  # (200, 2)
    # Generate some regular novel observations 生成一些常规的新奇观察
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations 产生一些异常新颖的观察
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    # fit the model 模型学习
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    print(n_error_train, n_error_test, n_error_outliers)



import numpy as np
from scipy import stats

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#随机数发生器
rng = np.random.RandomState(42)

# Example settings 示例设置
n_samples = 200
outliers_fraction = 0.25
clusters_separation = [0, 1, 2]

# define two outlier detection tools to be compared 定义两个异常的检测工具进行比较
classifiers = {
     "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                 kernel="rbf", gamma=0.1),
    "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
    "Isolation Forest": IsolationForest(max_samples=n_samples,
                                    contamination=outliers_fraction,
                                    random_state=rng),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 35,
                                           contamination=outliers_fraction)
           }

# Compare given classifiers under given settings 比较给定设置下的分类器
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:] = -1

# Fit the problem with varying cluster separation 将不同的集群分离拟合
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # Data generation 生成数据
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.r_[X1, X2]
    # Add outliers 添加异常值
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # Fit the model 模型拟合
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers 拟合数据和标签离群值
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            scores_pred = clf.negative_outlier_factor_
        else:
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            y_pred = clf.predict(X)
        threshold = stats.scoreatpercentile(scores_pred,
                                        100 * outliers_fraction)
        n_errors = (y_pred != ground_truth).sum()
        print(scores_pred)
        if clf_name == "Local Outlier Factor":
            # decision_function is private for LOF 决策函数是LOF的私有函数
            Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        print(Z)

