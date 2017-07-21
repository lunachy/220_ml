# coding:utf-8

import os
import time
import threading
import binascii
import multiprocessing
from math import sqrt
from multiprocessing import cpu_count
from queue import Queue
import cPickle

from PIL import Image
import numpy

WIDTHS = []
CPU_COUNT = cpu_count()
filename_list = []
length = 299
(DST_W, DST_H, SAVE_Q) = (length, length, 90)
root_dir = "/root/pe_classify/"
exe_path = os.path.join(root_dir, "virus")
pic_dir = os.path.join(root_dir, "pic_" + str(length))
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


def do_job(filename):
    # filename = args[0]
    img_name = os.path.basename(filename).split('.')[0]
    dst_img = os.path.join(pic_dir, img_name) + ".jpg"
    # print(filename, dst_img)
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = numpy.array([int(hexst[i: i + 2], 16) for i in range(0, len(hexst), 2)])  # 按字节分割
    width = int(sqrt(len(fh)))
    WIDTHS.append(width)
    fh = numpy.reshape(fh[:width * width], (-1, width))  # 根据设定的宽度生成矩阵
    fh = numpy.uint8(fh)

    im = Image.fromarray(fh)
    resize_img = im.resize((DST_W, DST_H), Image.ANTIALIAS)
    resize_img.save(dst_img, quality=SAVE_Q)


def list_files(path, depth=1):
    os.chdir(path)
    for obj in os.listdir(os.curdir):
        if os.path.isfile(obj):
            filename_list.append(os.getcwd() + os.sep + obj)
        if os.path.isdir(obj):
            if depth > 1:
                list_files(obj, depth - 1)
                os.chdir(os.pardir)


if __name__ == "__main__":
    cwd = os.getcwd()
    list_files(exe_path)
    os.chdir(cwd)

    start = time.time()
    print("Using %s cpu cores" % CPU_COUNT)
    # pool = multiprocessing.Pool(processes=CPU_COUNT, maxtasksperchild=400)
    # for i, filename in enumerate(filename_list):
    #     pool.apply_async(do_job, (filename,))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    #
    # pool.close()
    # pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    end = time.time()
    print("cost all time: %s seconds." % (end - start))

    for i, filename in enumerate(filename_list):
        do_job(filename)
        print i, filename

    print numpy.mean(WIDTHS)
    with open(os.path.join(root_dir, 'widths.dat'), 'wb') as f:
        cPickle.dump(WIDTHS, f)

    # with open(os.path.join(root_dir, 'widths.dat'), 'rb') as f:
    #     WIDTHS = cPickle.load(f)
    # w = numpy.array(WIDTHS)
    # print w.mean()
