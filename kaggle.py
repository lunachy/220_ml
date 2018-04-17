# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, Ridge
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from functools import partial
from gensim.models import Word2Vec
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# import nltk
# nltk.download('stopwords')

root_dir = '/root/chy'


def titanic():
    train_data = pd.read_csv(os.path.join(root_dir, 'titanic/train.csv'))
    test_data = pd.read_csv(os.path.join(root_dir, 'titanic/test.csv'))

    # select columns
    selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    X_train = train_data[selected_features]
    X_test = test_data[selected_features]
    Y_train = train_data['Survived']

    # fill na data
    X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
    X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
    X_train['Embarked'].fillna('S', inplace=True)
    X_test['Embarked'].fillna('S', inplace=True)
    X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
    # print X_train.info()
    # print X_test.info()

    dict_vec = DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='records'))
    X_test = dict_vec.transform(X_test.to_dict(orient='records'))
    # print dict_vec.feature_names_

    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    y_pred = rfc.predict(X_test)
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
    submission.to_csv(os.path.join(root_dir, 'titanic/rfc_results.csv'), index=False)

    y_test = pd.read_csv(os.path.join(root_dir, 'titanic/gender_submission.csv'))
    y_true = y_test['Survived']
    print accuracy_score(y_true, y_pred)

    rfc = RandomForestClassifier()
    params = {'n_estimators': range(10, 100, 10), 'max_depth': range(1, 10, 1)}
    gs = GridSearchCV(rfc, params, n_jobs=-1, cv=5, verbose=1)
    gs.fit(X_train, Y_train)
    print gs.best_params_
    print gs.best_score_


def review_to_text(review, remove_stopwords=True):
    # remove html label
    raw_text = BeautifulSoup(review, 'html.parser').get_text()
    # remove non-ascii
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    # remove stopwords if flag if true
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = filter(lambda w: w not in stop_words, words)
    return words


def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    raw_sentences = filter(lambda r_s: len(r_s) > 0, raw_sentences)
    review_to_text_p = partial(review_to_text, remove_stopwords=False)
    sentences = map(review_to_text_p, raw_sentences)
    return sentences


# 生成平均词向量
def make_avg_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features), dtype='float32')
    index2word_set = set(model.wv.index2word)
    for i, word in enumerate(words, 1):
        if word in index2word_set:
            feature_vec = np.add(feature_vec, model[word])
    return np.divide(feature_vec, i)


def imdb():
    train_data = pd.read_csv(os.path.join(root_dir, 'imdb/labeledTrainData.tsv'), delimiter='\t')
    test_data = pd.read_csv(os.path.join(root_dir, 'imdb/testData.tsv'), delimiter='\t')
    # print train_data.head()
    # print test_data.head()
    X_train = map(lambda r: ' '.join(review_to_text(r)), train_data['review'])
    X_test = map(lambda r: ' '.join(review_to_text(r)), test_data['review'])
    Y_train = train_data['sentiment']
    gs_count_flag, gs_tfidf_flag, gs_gbc_flag = True, True, True
    if gs_count_flag:
        # 使用Pipeline搭建两组使用朴素贝叶斯模型的分类器，特征提取分别为CountVectorizer，TfidfVectorizer
        pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
        # 分别设置用于模型超参数搜索的组合
        params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1, 1), (1, 2)],
                        'mnb__alpha': [0.1, 1.0, 10.0]}

        # 采用4折交叉验证的方法进行超参数搜索
        # 0.88216
        # {'mnb__alpha': 1.0, 'count_vec__binary': True, 'count_vec__ngram_range': (1, 2)}
        gs_count = GridSearchCV(pip_count, params_count, n_jobs=-1, cv=4, verbose=1)
        gs_count.fit(X_train, Y_train)
        print 'gs_count'
        print gs_count.best_score_
        print gs_count.best_params_

        count_y_predict = gs_count.predict(X_test)

    if gs_tfidf_flag:
        # 0.88712
        # {'tfidf_vec__ngram_range': (1, 2), 'tfidf_vec__binary': True, 'mnb__alpha': 0.1}
        pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
        params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1, 1), (1, 2)],
                        'mnb__alpha': [0.1, 1.0, 10.0]}
        gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, n_jobs=-1, cv=4, verbose=1)
        gs_tfidf.fit(X_train, Y_train)
        print 'gs_tfidf'
        print gs_tfidf.best_score_
        print gs_tfidf.best_params_

        tfidf_y_predict = gs_tfidf.predict(X_test)

    if gs_gbc_flag:
        unlabeled_train = pd.read_csv(os.path.join(root_dir, 'imdb/unlabeledTrainData.tsv'), delimiter='\t', quoting=3)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        corpora = []
        for review in unlabeled_train['review']:
            corpora += review_to_sentences(review.decode('utf-8'), tokenizer)

        num_features = 300
        model = Word2Vec(corpora, workers=4, size=num_features, min_count=20, window=10, sample=1e-3)
        model.init_sims(replace=True)
        # model_name = os.path.join(root_dir, 'imdb/imdb_word2vec_model')
        # model.save(model_name)
        # model = Word2Vec.load(model_name)

        make_avg_feature_vec_p = partial(make_avg_feature_vec, model=model, num_features=num_features)
        train_reviews = map(review_to_text, train_data['review'])
        train_data_vecs = map(make_avg_feature_vec_p, train_reviews)
        test_reviews = map(review_to_text, test_data['review'])
        test_data_vecs = map(make_avg_feature_vec_p, test_reviews)

        gbc = GradientBoostingClassifier()
        params_gbc = {'n_estimators': [10, 100, 500], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [2, 3, 4]}
        gs_gbc = GridSearchCV(gbc, params_gbc, n_jobs=-1, cv=4, verbose=1)
        gs_gbc.fit(train_data_vecs, Y_train)
        print 'gs_gbc'
        print gs_gbc.best_score_
        print gs_gbc.best_params_

        gbc_y_predict = gs_gbc.predict(test_data_vecs)


def digit():
    train_data = pd.read_csv(os.path.join(root_dir, 'digit/train.csv'))
    test_data = pd.read_csv(os.path.join(root_dir, 'digit/test.csv'))
    # print train_data.info()
    # print test_data.info()

    Y_train = np.array(train_data['label'])
    X_train = np.array(train_data.drop('label', 1))
    X_test = np.array(test_data)

    img_rows, img_cols = [28] * 2
    batch_size = 128
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    nb_classes = 10
    nb_epoch = 10

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    print('X_train shape:', X_train.shape)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, shuffle=True, validation_split=0.2)
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    # titanic()
    # imdb()
    digit()
