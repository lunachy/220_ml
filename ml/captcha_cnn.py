# coding=utf-8
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import random
import string
from keras.models import *
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model
from IPython.display import Image as Dis_Image
from PIL import Image
from scipy.misc import imread
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from tqdm import tqdm

K.clear_session()
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

characters = string.digits + string.ascii_uppercase
# print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
root_dir = "/root/captcha/"


# 定义数据生成器
def gen(batch_size=256):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for _ in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


# def generate_image(total_size=51200):
#     generator = ImageCaptcha(width=width, height=height)
#     for i in range(total_size):
#         random_str = ''.join([random.choice(characters) for j in range(4)])
#         generator.write(random_str, os.path.join(root_dir, 'image', random_str + '.png'))
#
#
# # gen_from_image
# def gen(batch_size=256):
#     for img in os.listdir(os.path.join(root_dir, 'image')):
#         X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
#         y = [np.zeros((batch_size, n_class), dtype=np.uint8) for _ in range(n_len)]
#         for i in range(batch_size):
#             X[i] = imread(os.path.join(root_dir, 'image', img), mode='RGB')
#             random_str = ''.join(os.path.splitext(img)[0])
#             for j, ch in enumerate(random_str):
#                 y[j][i, characters.find(ch)] = 1
#         yield X, y


# 解码
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


# 测试生成器
def test_gen_decode():
    X, y = next(gen(1))
    plt.imshow(X[0])
    plt.title(decode(y))


# 定义网络结构
def cnn():
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i in range(4):
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


# 网络结构可视化
def view_mode(model):
    plot(model, to_file="captcha_cnn_model.png", show_shapes=True)
    # Dis_Image('model.png')


# 测试模型
def test_model(model):
    X, y = next(gen(1))
    y_pred = model.predict(X)
    plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')


# 计算模型总体准确率
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num


# generate_image()

model = cnn()
view_mode(model)
# model.load_weights('captcha_cnn.h5')
# model.fit_generator(gen(), samples_per_epoch=1280, nb_epoch=100, validation_data=gen(), nb_val_samples=256)

# print('total_precision: %.2f%%' % (evaluate(model) * 100))

# 保存模型
# model.save('captcha_cnn.h5')


# images = os.listdir(os.path.join(root_dir, 'image'))
# X = np.zeros((1, height, width, 3), dtype=np.uint8)
# y = [np.zeros((len(images), n_class), dtype=np.uint8) for _ in range(n_len)]
# random_strs = []
# for i, img in enumerate(images):
#     X[0] = imread(os.path.join(root_dir, 'image', img), mode='RGB')  # .resize((height, width, 3))
#     random_str = ''.join(os.path.splitext(img)[0])
#     y_pred = model.predict(X)
#     print random_str, decode(y_pred)

# y_pred = model.predict(X)
# print decode(y_pred)
# # print y_pred
# for i, y in enumerate(y_pred):
#     print decode(y), random_strs[i]

# img = "0017.jpg"
# X = imread(os.path.join(root_dir, 'image', img), mode='RGB')
# print X.shape
# im = Image.fromarray(X)
# X = im.resize((width, height), Image.ANTIALIAS)
# print im.size
# X.save('/tmp/0017.jpg')
# print X.size
# X = np.expand_dims(X, 0)
# print X.shape
# random_str = ''.join(os.path.splitext(img)[0])
# y_pred = model.predict(X)
# print('real: %s\npred: %s' % (random_str, decode(y_pred)))


# generator = ImageCaptcha(width=width, height=height)
# random_str = ''.join([random.choice(characters) for j in range(4)])
# # random_str = 'O0O0'
# X = generator.generate_image(random_str)
# X = np.expand_dims(X, 0)
#
# y_pred = model.predict(X)
# print('real: %s\npred: %s' % (random_str, decode(y_pred)))

K.clear_session()

