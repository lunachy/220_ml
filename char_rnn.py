# coding=utf-8
import os
import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils


raw_text = open('/data/chy/alice.txt').read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
print("chars length: %s, raw_text_length: %s" % (len(chars), len(raw_text)))

seq_length = 100
x = []
y = []
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

n_patterns = len(x)
n_vocab = len(chars)

# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# 简单normal到0-1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)

model_path = '/data/chy/char_cnn_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = Sequential()
    model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# save_best_callback = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=False)
model.fit(x, y, epochs=100, batch_size=4096, callbacks=[EarlyStopping(monitor="loss", patience=5)])
model.save(model_path)


def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input) - seq_length):]:
        res.append(char_to_int[c])
    return res


def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c


def generate_article(init, rounds=500):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = """Either the well was very deep, or she fell very slowly, for she had
plenty of time as she went down to look about her and to wonder what was"""
article = generate_article(init)
print(article)
