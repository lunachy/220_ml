# coding=utf-8
import time
import os
import signal
from multiprocessing import cpu_count, Pool

CPU_COUNT = cpu_count()


def do_job():
    os.system("curl www.163.com")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    start = time.time()
    print("Using %s cpu cores" % CPU_COUNT)
    pool = Pool(processes=CPU_COUNT, initializer=init_worker, maxtasksperchild=400)
    while True:
        pool.apply_async(do_job)

    pool.close()
    pool.join()
    end = time.time()
    print("cost all time: %s seconds." % (end - start))
