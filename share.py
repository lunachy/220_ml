# coding=utf-8

from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import time
import sys
import signal
import tushare as ts

DELTA_DATE = 5
LIMIT = 9.9


def get_start_date(day_delta):
    end_date = datetime.now()
    delta = timedelta(days=day_delta)
    start = (end_date - delta).strftime('%Y-%m-%d')
    return start


start = get_start_date(DELTA_DATE + 1)


# 两连扳
def limit_up_2(share):
    data = ts.get_hist_data(share, '2017-05-01')
    try:
        dict_data = data.to_dict('list')
        all_data = data.to_dict('dict')
    except AttributeError:
        return

    open = dict_data["open"]
    high = dict_data["high"]
    low = dict_data["low"]
    volume = dict_data["volume"]
    close = dict_data["close"]
    p_change = dict_data["p_change"]
    flag = True
    for i in range(len(high) - 1):
        if (high[i] * (100.0 + p_change[i]) / close[i] - 100.0) > LIMIT and (
                high[i + 1] * (100.0 + p_change[i + 1]) / close[i + 1] - 100.0) > LIMIT and not (
                low[i] == high[i] and low[i + 1] == high[i + 1]):
            volume_values = all_data["volume"].values()
            date = all_data["volume"].keys()[volume_values.index(volume[i + 1])]
            return share, date
    return ''


# 连续两天涨停，第三天跌停
def up_fall(share):
    data = ts.get_hist_data(share, '2016-01-01', '2017-05-01')
    try:
        dict_data = data.to_dict('list')
        all_data = data.to_dict('dict')
    except AttributeError:
        return

    open = dict_data["open"]
    high = dict_data["high"]
    low = dict_data["low"]
    volume = dict_data["volume"]
    close = dict_data["close"]
    p_change = dict_data["p_change"]
    for i in range(len(high) - 2):
        if high[i] == close[i] and p_change[i] < -LIMIT and high[i + 1] == close[i + 1] and p_change[i + 1] > LIMIT and \
                high[i + 2] == close[i + 2] and p_change[i + 2] > LIMIT:
            volume_values = all_data["volume"].values()
            date = all_data["volume"].keys()[volume_values.index(volume[i])]
            return share, date
    return ''


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    dict_shares = ts.get_stock_basics().name.to_dict()
    all_shares = dict_shares.keys()
    t1 = time.time()
    pool = Pool(cpu_count(), initializer=init_worker, maxtasksperchild=400)
    _shares = pool.map(up_fall, all_shares)
    shares = filter(lambda x: x, _shares)
    t2 = time.time()
    print "cost {} seconds".format(t2 - t1)
    print len(shares)
    print shares

# In [8]: ts.get_hist_data('300647', '2018-06-07')
# Out[8]:
#              open   high  close    low     volume  price_change  p_change  \
# date
# 2018-06-20  16.45  20.11  20.11  16.45  304829.75          1.83     10.01
# 2018-06-19  18.28  18.28  18.28  18.28    7713.00         -2.03     -9.99
# 2018-06-08  17.84  20.31  20.31  17.11  334761.41          1.85     10.02
# 2018-06-07  16.80  18.46  18.46  16.18  372038.69          1.68     10.01
#
#                ma5    ma10    ma20      v_ma5     v_ma10     v_ma20
# date
# 2018-06-20  18.788  18.342  20.536  205834.14  136840.06  102600.98
# 2018-06-19  17.816  18.779  20.622  149981.59  117686.31   89940.47
# 2018-06-08  16.932  19.236  20.805  159691.37  123274.45   91482.80
# 2018-06-07  15.390  19.555  21.009  117239.03   97009.87   78167.64
#
# [4 rows x 14 columns]

#
# Definition: data.to_dict(self, outtype='dict')
# Docstring:
# Convert DataFrame to dictionary.
#
# Parameters
# ----------
# outtype : str {'dict', 'list', 'series', 'records'}
#     Determines the type of the values of the dictionary. The
#     default `dict` is a nested dictionary {column -> {index -> value}}.
#     `list` returns {column -> list(values)}. `series` returns
#     {column -> Series(values)}. `records` returns [{columns -> value}].
#     Abbreviations are allowed.
#
#
# Returns
# -------
# result : dict like {column -> {index -> value}}
