# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state=9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
print y_pred
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print metrics.calinski_harabaz_score(X, y_pred)
