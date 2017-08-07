# coding=utf-8
import string

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.models import *
from scipy.misc import imread

K.clear_session()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

characters = string.digits + string.ascii_uppercase
# print(characters)

width, height, n_len, n_class = 100, 33, 4, len(characters)
root_dir = "/root/captcha/"
train_dir = "/root/captcha/code_pic"
test_dir = "/root/captcha/test_pic"
nb_epoch = 50
batch_size = 64


# gen_from_image dir
def gen(pic_dir):
    os.chdir(pic_dir)
    images = os.listdir(pic_dir)
    X = np.zeros((len(images), height, width, 3), dtype=np.uint8)
    y = [np.zeros((len(images), n_class), dtype=np.uint8) for _ in range(n_len)]
    for i, img in enumerate(images):
        X[i] = imread(img, mode='RGB')
        random_str = ''.join(os.path.splitext(img)[0])
        for j, ch in enumerate(random_str):
            y[j][i, characters.find(ch)] = 1
    return X, y


# 定义网络结构
def cnn():
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i in range(2):
        x = Conv2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = Conv2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


# 解码
def decode(y):
    decode_y = []
    for i in range(y[0].shape[0]):
        _y = np.argmax(np.array(y), axis=2)[:, i]
        decode_y.append(''.join([characters[x] for x in _y]))
    return decode_y


# 批量识别有标签验证码
def recognize_batch_captcha(model, pic_dir):
    x, y = gen(pic_dir)
    y_pred = model.predict(x)
    return decode(y), decode(y_pred)


# 识别单个验证码
def recognize_single_captcha(pic_path):
    model = load_model(os.path.join(root_dir, 'captcha_gdrc_233_50.h5'))
    x = np.zeros((1, height, width, 3), dtype=np.uint8)
    x[0] = imread(pic_path, mode='RGB')
    y_pred = model.predict(x)
    return decode(y_pred)[0]


# 计算模型总体准确率
def evaluate(model, pic_dir):
    x, y = gen(pic_dir)
    y_pred = model.predict(x)
    return np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))


# model = cnn()
# view_mode(model)
# x, y = gen(train_dir)
# model = load_model(os.path.join(root_dir, 'captcha_gdrc_233_50.h5'))
# model.fit(x, y, validation_split=0.1, epochs=nb_epoch, batch_size=batch_size)
# accu = evaluate(model, test_dir)
# print('total_precision: %.2f%%' % (accu * 100))

# y1, y_pred = recognize_batch_captcha(model, test_dir)
# for _y, _ypred in zip(decode(y1), decode(y_pred)):
#     print _y, _ypred
# single_y = recognize_single_captcha('/root/captcha/raw_pic/1501741585126195.jpg')
# print single_y
# 保存模型
# model.save(os.path.join(root_dir, 'captcha_gdrc_322_50.h5'))

K.clear_session()
