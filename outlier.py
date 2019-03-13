# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import Counter
from sklearn.preprocessing import LabelEncoder


def datetime2int(datetime):
    return int(datetime.split(' ')[1].split(':')[0])


def detect_outliers(df, n, features):
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


class_label = LabelEncoder()
df = pd.read_csv("/data/sample.txt", header=0)
# columns = df.columns.values
df['hour'] = df[r'交易时间'].apply(datetime2int)
df["qudao"] = class_label.fit_transform(df["渠道"].values)
df["diqu"] = class_label.fit_transform(df["发起方所处地区"].values)

Outliers_to_drop = detect_outliers(df, 1, ['交易金额', '发起方id', '发起方年龄', '接收方ID', 'hour', 'qudao', 'diqu'])
df.iloc[Outliers_to_drop, :].to_csv('/data/outlier.txt')
# print(df.iloc[Outliers_to_drop,:])
# Drop outliers
# df = df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# use IsolationForest
# rng = np.random.RandomState(42)
# # fit the model
# clf = IsolationForest(max_samples=100, random_state=rng, contamination='auto')
# clf.fit(df)   # or use k-fold
# y_pred_train = clf.predict(df)
# print(y_pred_train)
