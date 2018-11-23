# -*- coding: utf-8 -*-
import os
from collections import Counter
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


def get_dga_ml_features(x, alexa_vocab, alexa_counts, word_vocab, word_counts):
    alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0, vocabulary=alexa_vocab)
    word_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0, vocabulary=word_vocab)
    x = x.split('.')[0].strip().lower()
    x_length = len(x)
    x_entropy = entropy(x)
    x_alexa_grams = alexa_counts * alexa_vc.transform([x]).T
    x_word_grams = word_counts * word_vc.transform([x]).T
    x_diff = x_alexa_grams.item() - x_word_grams.item()
    features = np.array([x_length, x_entropy, x_alexa_grams.item(), x_word_grams.item(), x_diff]).reshape(1, -1)
    return features


def load_models(root_dir):
    clf_gb = joblib.load(os.path.join(root_dir, 'models', 'domain_dga_GradientBoosting.pkl'))
    alexa_vocab = joblib.load(os.path.join(root_dir, 'counts', 'domain_dga_vocab_alexa.txt'))
    alexa_counts = np.load(os.path.join(root_dir, 'counts', 'domain_dga_alexa_counts.npz'))['alexa_counts']
    word_vocab = joblib.load(os.path.join(root_dir, 'counts', 'domain_dga_vocab_word.txt'))
    word_counts = np.load(os.path.join(root_dir, 'counts', 'domain_dga_word_counts.npz'))['dict_counts']
    return clf_gb, alexa_vocab, alexa_counts, word_vocab, word_counts


def run_predict_ml(x):
    current_path = os.path.dirname(os.path.abspath(__file__))
    clf_gb, alexa_vocab, alexa_counts, word_vocab, word_counts = load_models(current_path)
    x_ml_features = get_dga_ml_features(x, alexa_vocab, alexa_counts, word_vocab, word_counts)
    y_pred_gb = clf_gb.predict(x_ml_features)[0]
    return True if y_pred_gb == 'dga' else False


if __name__ == "__main__":
    domain = "detectportal.firefox.com.edgesuite.net"
    print run_predict_ml(domain)  # dga or legal
