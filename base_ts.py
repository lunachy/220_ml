# coding=utf-8
import sys

sys.path.append('/hadoop2/asap/ssa/python_package')
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyflux as pf
import statsmodels.tsa.stattools as st


def create_diffed_ts(df, diffn):
    if diffn != 0:
        df['diff'] = df[df.columns[0]].apply(lambda x: float(x)).diff(diffn)
    else:
        df['diff'] = df[df.columns[0]].apply(lambda x: float(x))
    df.drop(df.iloc[:diffn].index, inplace=True)
    return df


def prediction_recover_p(predicitons, df, diffn, test_size):
    if diffn != 0:
        shift = df[df.columns[0]].shift(diffn - test_size)
        predicitons = predicitons + shift[-test_size:].values
    return predicitons


def choose_order(ts, maxar, maxma):
    order = st.arma_order_select_ic(ts, maxar, maxma, ic=['aic', 'bic', 'hqic'])
    return order.bic_min_order


def rolling_forecast_pf(train, order, days):
    model = pf.ARIMA(data=train, ar=order[0], ma=order[1], target="diff", family=pf.Normal())
    model.fit("MLE")
    output = model.predict(days, intervals=True)
    predictions = [np.int(x) for x in output.iloc[:, 0]]
    upperbound = np.int(output.iloc[0, 4])
    lowerbound = np.int(output.iloc[0, 1])
    return predictions, upperbound, lowerbound


def run_ariam_pf(df, maxar, maxma, diffn, test_size):
    train = df.copy()
    train = create_diffed_ts(train, diffn)
    order = choose_order(train["diff"], maxar, maxma)
    predicitons, upperbound, lowerbound = rolling_forecast_pf(train, order, test_size)
    predictions_recover = prediction_recover_p(predicitons, train, diffn, test_size)
    upperbound = prediction_recover_p(upperbound, train, diffn, 1)
    lowerbound = prediction_recover_p(lowerbound, train, diffn, 1)
    print("done")
    return predictions_recover, lowerbound, upperbound


def get_first_week(data, test_size, multiplier):
    a = data["Number"]
    predictions = np.repeat(np.median(a), test_size)
    predictions = map(lambda x: np.int(x), predictions)
    mad = np.median(np.abs((a - np.median(a))))
    lowerbound = np.int(np.median(a) - multiplier * mad)
    lowerbound = 0 if lowerbound < 0 else lowerbound
    upperbound = np.int(np.median(a) + multiplier * mad)
    return predictions, lowerbound, upperbound


def get_first_month(data, test_size, multiplier):
    last_7 = data.sort_values("Date")[-7:]["Number"] == 0
    if not False in list(last_7):
        new_predictions, lowerbound, upperbound = get_first_week(data, test_size, multiplier=1)
    else:
        data["Number"] = data["Number"].apply(lambda x: x + 1 if x == 0 else x)
        data["Date"] = pd.to_datetime(data["Date"])
        daytype = set(map(lambda x: x.weekday(), data["Date"]))
        data["weekday"] = map(lambda x: x.weekday(), data["Date"])
        weekday_ratio = {}
        baseline = sorted(list(data["Date"]))[-1].weekday()
        for _daytype in daytype:
            ratio = np.mean(data[data["weekday"] == _daytype]["Number"]) * 1.0 / np.mean(
                data[data["weekday"] == baseline]["Number"])
            weekday_ratio[_daytype] = ratio
        order = range(baseline + 1, 7) + range(0, baseline + 1)
        new_ratio = [weekday_ratio[daytype] for daytype in order]
        new_ratio = [1 if x != x else x for x in new_ratio]
        new_ratio = new_ratio[0:test_size]
        predictions, lowerbound, upperbound = get_first_week(data, test_size, multiplier)
        try:
            new_predictions = [np.int(x) for x in new_ratio * predictions]
        except:
            new_predictions = map(lambda x: np.int(x), predictions)
    lowerbound = 0 if lowerbound < 0 else lowerbound
    return new_predictions, lowerbound, upperbound


def check_status(threatNumber, lowerbound, upperbound):
    if threatNumber < lowerbound:
        status = -1
    elif threatNumber > upperbound:
        status = 1
    else:
        status = 0
    return status


def check_stability(array, value):
    new_array = []
    for i, item in enumerate(array):
        if i > 1 and (item <= 0 or item >= 3 * value or item <= (1 / 3) * value):
            item = value
        else:
            pass
        new_array.append(item)
    return new_array


def get_Predictions_types(data, diffn, test_size, maxar=5, maxma=5, multiplier=2):
    Predictions = []
    if data.shape[0] != 0:
        types = set(data["Type"])
        for _type in types:
            print(_type)
            data_1 = data[data["Type"] == _type]
            data_2 = data_1[data_1["Number"] == -1]
            if data_1.shape[0] == data_2.shape[0]:
                pass
            else:
                begin_date = sorted(data_1[data_1["Number"] != -1]["Date"])[0]
                data_s = data_1[data_1["Date"] >= begin_date]
                a = [(x is None) or (x != x) for x in data_s["Predict1"]]
                dates = sorted(list(data_s[a]["Date"]))
                print(dates)
                for date in dates:
                    df = data_s[data_s["Date"] <= date][["Date", "Number"]].sort_values("Date")
                    if df[df["Number"] == 0].shape[0] == df.shape[0]:
                        lowerbound = 0
                        upperbound = 0
                        new_lowerbound = None
                        new_upperbound = None
                        predictions = [0] * test_size
                        predictions.extend([lowerbound, upperbound, new_lowerbound, new_upperbound, date, _type])
                    else:
                        days = df.shape[0]
                        if days <= 7:
                            print("use first_week method")
                            # threatNumber = df[df["Date"] == date]["Number"].item()
                            df = df[df["Number"] != -1]
                            predictions, lowerbound, upperbound = get_first_week(df, test_size, multiplier)
                            predictions = list(predictions)
                            # status = check_status(threatNumber, lowerbound, upperbound)
                            new_lowerbound = None
                            new_upperbound = None
                            predictions.extend([lowerbound, upperbound, new_lowerbound, new_upperbound, date, _type])
                        elif (days > 7) & (days <= 40):
                            print("use first_month method")
                            # threatNumber = df[df["Date"] == date]["Number"].item()
                            df = df[df["Number"] != -1]
                            predictions, lowerbound, upperbound = get_first_month(df, test_size, multiplier)
                            predictions = list(predictions)
                            # status = check_status(threatNumber, lowerbound, upperbound)
                            new_lowerbound = None
                            new_upperbound = None
                            predictions.extend([lowerbound, upperbound, new_lowerbound, new_upperbound, date, _type])
                        else:
                            try:
                                print("use time series method")
                                df.set_index('Date', inplace=True)
                                df["Number"] = df["Number"].apply(lambda x: np.nan if x == -1 else x)
                                df = df.interpolate(method='time')
                                threatNumber = df.loc[date].item()
                                predictions, new_lowerbound, new_upperbound = run_ariam_pf(df, maxar, maxma, diffn,
                                                                                           test_size)
                                new_lowerbound = 0 if new_lowerbound < 0 else new_lowerbound
                                # status = check_status(threatNumber, lowerbound, upperbound)
                                predictions = list(predictions)
                                predictions = check_stability(predictions, threatNumber)
                                try:
                                    yday = list(data_1[data_1["Date"] == date - timedelta(days=1)].values[0])
                                    lowerbound = yday[-2]
                                    upperbound = yday[-1]
                                except:
                                    lowerbound = new_lowerbound
                                    upperbound = new_upperbound
                                if (lowerbound is None) or (lowerbound != lowerbound):
                                    lowerbound = new_lowerbound
                                if (upperbound is None) or (upperbound != upperbound):
                                    upperbound = new_upperbound
                                predictions.extend(
                                    [lowerbound, upperbound, new_lowerbound, new_upperbound, date, _type])
                            except ValueError:
                                print("use back up method")
                                yday = list(data_1[data_1["Date"] == date - timedelta(days=1)].values[0])
                                predictions = yday[7:13]
                                new_lowerbound = yday[-2]
                                new_upperbound = yday[-1]
                                lowerbound = yday[-2]
                                upperbound = yday[-1]
                                # status = check_status(threatNumber, lowerbound, upperbound)
                                predictions.extend(
                                    [yday[12], lowerbound, upperbound, new_lowerbound, new_upperbound, date, _type])
                    Predictions.append(predictions)
    return Predictions


def get_Predictions_types_wo_bound(data, diffn, test_size, maxar=5, maxma=5, multiplier=2):
    Predictions = []
    if data.shape[0] != 0:
        types = set(data["Type"])
        for _type in types:
            print(_type)
            data_1 = data[data["Type"] == _type]
            data_2 = data_1[data_1["Number"] == -1]
            if data_1.shape[0] == data_2.shape[0]:
                pass
            else:
                begin_date = sorted(data_1[data_1["Number"] != -1]["Date"])[0]
                data_s = data_1[data_1["Date"] >= begin_date]
                a = [(x is None) or (x != x) for x in data_s["Predict1"]]
                dates = sorted(list(data_s[a]["Date"]))
                print(dates)
                for date in dates:
                    df = data_s[data_s["Date"] <= date][["Date", "Number"]].sort_values("Date")
                    if df[df["Number"] == 0].shape[0] == df.shape[0]:
                        predictions = [0] * test_size
                        # predictions.extend([date, _type])
                        predictions = [str(predictions), date, _type]
                    else:
                        days = df.shape[0]
                        if days <= 7:
                            print("use first_week method")
                            # threatNumber = df[df["Date"] == date]["Number"].item()
                            df = df[df["Number"] != -1]
                            predictions, _, _ = get_first_week(df, 7, multiplier)
                            predictions = list(predictions)
                            predictions.extend([0] * (test_size - 7))
                            # status = check_status(threatNumber, lowerbound, upperbound)
                            # predictions.extend([date, _type])
                            predictions = [str(predictions), date, _type]
                        elif (days > 7) & (days <= 40):
                            print("use first_month method")
                            # threatNumber = df[df["Date"] == date]["Number"].item()
                            df = df[df["Number"] != -1]
                            predictions, _, _ = get_first_month(df, 7, multiplier)
                            predictions = list(predictions)
                            predictions.extend([0] * (test_size - 7))
                            # status = check_status(threatNumber, lowerbound, upperbound)
                            # predictions.extend([date, _type])
                            predictions = [str(predictions), date, _type]
                        else:
                            try:
                                print("use time series method")
                                df.set_index('Date', inplace=True)
                                df["Number"] = df["Number"].apply(lambda x: np.nan if x == -1 else x)
                                df = df.interpolate(method='time')
                                threatNumber = df.loc[date].item()
                                predictions, _, _ = run_ariam_pf(df, maxar, maxma, diffn, test_size)
                                # status = check_status(threatNumber, lowerbound, upperbound)
                                predictions = list(predictions)
                                predictions = check_stability(predictions, threatNumber)
                                # predictions.extend([date, _type])
                                predictions = [str(predictions), date, _type]
                            except ValueError:
                                print("use back up method")
                                yday = list(data_1[data_1["Date"] == date - timedelta(days=1)].values[0])
                                predictions = yday[5]
                                # status = check_status(threatNumber, lowerbound, upperbound)
                                # predictions.extend([date, _type])
                                predictions = [str(predictions), date, _type]
                    Predictions.append(predictions)
    return Predictions
