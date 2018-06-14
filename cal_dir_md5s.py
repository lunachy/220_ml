# -*- coding: utf-8 -*-

import os
import sys
import hashlib
import time
import magic
import signal
import logging.handlers
from ConfigParser import RawConfigParser
from multiprocessing import Pool, cpu_count

FILE_CHUNK_SIZE = 16 * 1024
FILTER_FORMAT = ['application/x-dosexec', 'text/x-python', 'text/x-shellscript', 'text/x-msdos-batch']
CPU_COUNT = cpu_count()
EXCEPT_DIR = ['run']
log = logging.getLogger(__file__[:-3])


def init_logging(log_file):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    fh = logging.handlers.WatchedFileHandler(log_file)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.setLevel(logging.INFO)


def md5_clean(filepath):
    md5s = set()
    with open(filepath, 'r') as f1:  # 237000736 items
        with open('all_md5_set.txt', 'w') as f2:  # 29967148 items
            for line in f1:
                _md5 = line.strip()
                if _md5 not in md5s and len(_md5) == 32 and not _md5.startswith('#'):
                    f2.write(_md5 + '\n')
                md5s.add(_md5)


def read_all_md5(filepath):
    md5s = []
    with open(filepath, 'r') as f1:
        for line in f1:
            _md5 = line.strip()
            md5s.append(_md5)
    return md5s


def get_chunks(file_path):
    """Read file contents in chunks (generator)."""
    with open(file_path, "rb") as fd:
        while True:
            chunk = fd.read(FILE_CHUNK_SIZE)
            if not chunk: break
            yield chunk


def compare(file_path, timestamp):
    if magic.from_file(file_path, mime=True) in FILTER_FORMAT and os.path.getmtime(
            file_path) > timestamp:
        md5 = hashlib.md5()
        for chunk in get_chunks(file_path):
            md5.update(chunk)
        _md5 = md5.hexdigest()
        data = _md5 + ' ' + file_path + '\n'
        if _md5 in md5s:
            log.info('match file, %s', _md5 + ' ' + file_path)
        return data
    return None


def calc_md5(file_path, timestamp):
    if magic.from_file(file_path, mime=True) in FILTER_FORMAT and os.path.getmtime(
            file_path) > timestamp:
        md5 = hashlib.md5()
        for chunk in get_chunks(file_path):
            md5.update(chunk)
        data = md5.hexdigest() + ' ' + file_path + '\n'
        return data
    else:
        return None


def to_file(data):
    if data:
        with open('malware_md5.txt', 'a') as f:
            f.write(data)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_md5s(root_dir, mode='increment'):
    if mode != 'increment':
        dt = '2000-01-01'
        timeArray = time.strptime(dt, "%Y-%m-%d")
        timestamp = time.mktime(timeArray)
    else:
        with open('md5_date.txt') as f:
            dt = f.readline().strip()
            timeArray = time.strptime(dt, "%Y-%m-%d")
            timestamp = time.mktime(timeArray)

    pool = Pool(processes=CPU_COUNT - 2, initializer=init_worker, maxtasksperchild=400)

    cnt = 0
    cnt_tmp = 0
    _i = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # if dirpath.split('/')[1] in EXCEPT_DIR:
        #     continue
        for _i, _filename in enumerate(filenames, 1):
            abs_filename = os.path.join(dirpath, _filename)
            print abs_filename
            pool.apply_async(compare, (abs_filename, timestamp), callback=to_file)
        cnt += _i
        if cnt - cnt_tmp > 500000:
            cnt_tmp = cnt
            log.info('current file cnt: %s', cnt)
    pool.close()
    pool.join()


if __name__ == '__main__':
    init_logging('/root/chy/cal_dir_md5s.log')
    a = time.time()
    # md5_clean('all.md5')
    md5s = read_all_md5('all_md5_set.txt')
    b = time.time()
    log.info('reading md5s costs %s seconds', int(b - a))
    root_dir = '/run'
    get_md5s(root_dir, mode='full')
    c = time.time()
    log.info('matching md5s costs %s seconds', int(c - b))
