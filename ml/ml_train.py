from __future__ import print_function
import argparse
import json
import os
import logging.handlers
import time
from ml_classification import ml_classification
from ml_regression import ml_regression
from ml_cluster import ml_cluster


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
    parser.add_argument("--model_parameters", default={}, help="", type=str)
    parser.add_argument("--train_nickname", default="test", help="", type=str)
    return parser


def ml_train(data_input_path, features, model_parameters, model_id, train_id, train_nickname):
    if model_id.startswith("1"):
        ml_classification(data_input_path, features, model_parameters, model_id, train_id)
    if model_id.startswith("2"):
        ml_regression(data_input_path, features, model_parameters, model_id, train_id)
    if model_id.startswith("3"):
        ml_cluster(data_input_path, features, model_parameters, model_id, train_id)


if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    time_format = "%Y-%m-%d  %H:%M:%S"
    logging.info('start time: %s !', time.strftime(time_format, time.localtime()))
    ml_train(options.data_input_path, options.features, options.model_parameters, options.model_id, options.train_id, options.train_nickname)
    logging.info('finish time: %s !', time.strftime(time_format, time.localtime()))




