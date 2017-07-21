# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.layers import Input
from keras.models import model_from_json
import os
import codecs
import json
import sys
import numpy as np

import scipy.misc
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))


def get_topk_data(pic_dir, label, img_rows, img_cols, k=10):
    def read_image(md5s):
        X_inner = np.zeros([len(md5s), img_rows, img_cols])
        for i, md5 in enumerate(md5s):
            image = os.path.join(pic_dir, "VirusShare_" + md5 + ".jpg")
            mat = scipy.misc.imread(image).astype(np.float32)
            X_inner[i, :, :] = mat
        return X_inner

    types = set(label["type"])
    X = np.zeros([len(types) * k, img_rows, img_cols])
    Y = []
    for i, type in enumerate(types):
        data = label.loc[label["type"] == type].head(k)
        Y += [type] * k
        mat = read_image(data["md5"])
        X[i * k: (i + 1) * k, :, :] = mat
    Y = np.asarray(Y, dtype=np.str)
    return X, Y


def get_data(data_set, label, img_rows, img_cols):
    X = np.zeros([label.shape[0], img_rows, img_cols])
    for cc, x in enumerate(label["md5"]):
        image = os.path.join(data_set, "VirusShare_" + x + ".jpg")
        mat = scipy.misc.imread(image).astype(np.float32)
        X[cc, :, :] = mat
    return X


def save_model(model):
    with codecs.open(model_path, "w", "utf-8") as json_file:
        json.dump(model.to_json(), json_file, encoding="utf-8")  # , ensure_ascii=False, sort_keys=False, indent=4)

    # serialize weights to HDF5
    model.save_weights(weights_path)
    print("Saved model to disk")


def load_model():
    with codecs.open(model_path, "r", "utf-8") as json_file:
        json_string = json.load(json_file, encoding="utf-8")
    model = model_from_json(json_string)
    model.load_weights(weights_path)
    return model


length = 299
nb_epoch = 100
batch_size = 32
root_dir = "/root/pe_classify"
img_rows, img_cols = length, length
input_shape = (img_rows, img_cols, 1)
model_path = os.path.join(root_dir, "model.json")
weights_path = os.path.join(root_dir, "weights.h5")

label = pd.read_csv(os.path.join(root_dir, "train_label.csv"))
nb_classes = len(set(label["type"]))

x = get_data(os.path.join(root_dir, "pic_" + str(length)), label, img_rows, img_cols)
y = pd.factorize(label["type"])[0]
# x, y = get_topk_data(os.path.join(root_dir, "pic_" + str(length)), label, img_rows, img_cols)
# y = pd.factorize(y)[0]
print("nb_classes:", nb_classes)
print("original x shape:", x.shape)
print("original y shape:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


print("x_train shape:", x_train.shape)
print("x_test  shape", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test  shape", y_test.shape)

input_tensor = Input(shape=(img_rows, img_cols, 1))

model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=True, classes=nb_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.2, nb_epoch=nb_epoch, batch_size=batch_size)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])

save_model(model)
# model = load_model()

predict_labels = model.predict(x_test)
y_pred = np.argmax(predict_labels, 1)
y_test = np.argmax(y_test, 1)

print("classification_report:\n", classification_report(y_test, y_pred))
print("accuracy_score:\n", accuracy_score(y_test, y_pred))
print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
