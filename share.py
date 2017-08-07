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


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    dict_shares = ts.get_stock_basics().name.to_dict()
    all_shares = dict_shares.keys()
    t1 = time.time()
    pool = Pool(cpu_count(), initializer=init_worker, maxtasksperchild=400)
    _shares = pool.map(limit_up_2, all_shares)
    shares = filter(lambda x: x, _shares)
    t2 = time.time()
    print "cost {} seconds".format(t2 - t1)
    print len(shares)
    print shares

# open   high  close    low     volume  price_change  p_change  \
# date
# 2016-06-21  17.88  17.95  17.81  17.75  158528.73         -0.04     -0.22
# 2016-06-20  17.93  17.93  17.86  17.75   96814.95          0.09      0.51
# 2016-06-17  17.75  17.99  17.79  17.74  174122.33         -0.01     -0.06
# 2016-06-16  17.67  17.93  17.80  17.60  308711.28          0.06      0.34
#
#                ma5    ma10    ma20      v_ma5     v_ma10     v_ma20  turnover
# date
# 2016-06-21  17.804  17.863  17.865  200299.81  207123.38  187827.34      0.08
# 2016-06-20  17.802  17.884  17.847  206140.78  209274.02  185471.22      0.05
# 2016-06-17  17.784  17.901  17.822  246880.96  217379.05  185820.88      0.09
# 2016-06-16  17.826  17.942  17.807  259573.09  218279.24  192595.99      0.17
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
