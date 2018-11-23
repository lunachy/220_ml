# coding=utf-8
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import sys
import random
import string
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot
from IPython.display import Image as Dis_Image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from tqdm import tqdm

K.clear_session()
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

characters = string.digits + string.ascii_uppercase
# print(characters)

width, height, n_len, n_class = 100, 33, 4, len(characters) + 1


# 定义 CTC Loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# 解码
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


# 定义网络结构
rnn_size = 128
input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
# 17 * 6 * 32
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
model.load_weights('captcha_ctc.h5')


# 网络结构可视化
def view_mode(model):
    plot(model, to_file="captcha_ctc_model.png", show_shapes=True)
    Dis_Image('captcha_ctc_model.png')
    # SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# 定义数据生成器
def gen(batch_size=256):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        generator = ImageCaptcha(width=width, height=height)
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size) * int(conv_shape[1] - 2), np.ones(batch_size) * n_len], np.ones(batch_size)


# 测试生成器
def test_gen_decode():
    [X_test, y_test, _, _], _ = next(gen(1))
    plt.imshow(X_test[0].transpose(1, 0, 2))
    plt.title(''.join([characters[x] for x in y_test[0]]))


# 测试模型
def test_model(model):
    characters2 = characters + ' '
    [X_test, y_test, _, _], _ = next(gen(1))
    y_pred = base_model.predict(X_test)
    y_pred = y_pred[:, 2:, :]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([characters[x] for x in out[0]])
    y_true = ''.join([characters[x] for x in y_test[0]])

    plt.imshow(X_test[0].transpose(1, 0, 2))
    plt.title('pred:' + str(out) + '\ntrue: ' + str(y_true))

    argmax = np.argmax(y_pred, axis=2)[0]
    list(zip(argmax, ''.join([characters2[x] for x in argmax])))


# 计算模型总体准确率
def evaluate(model, batch_num=10):
    batch_acc = 0
    generator = gen(128)
    for i in range(batch_num):
        [X_test, y_test, _, _], _ = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    return batch_acc / batch_num


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print
        print 'acc: %f%%' % acc


evaluator = Evaluate()
view_mode(model)
sys.exit()
model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=200,
                    callbacks=[EarlyStopping(patience=5), evaluator],
                    validation_data=gen(), nb_val_samples=1280)
total_precision = evaluate(model)
print(total_precision)

# 保存模型
model.save('captcha_ctc.h5')
K.clear_session()
