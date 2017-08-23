# =============================================================================
#
# Copyright 2017 Carlos Alberto Chavez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA, ARIMA, AR
import statsmodels.tsa.seasonal as snl
import time


def remove_trend_and_seasonality(data):
    """
    Removes trend and seasonality by decomposing a dataset
    :param data:
    :return:
    """
    # Decomposing
    decomposition = snl.seasonal_decompose(data, freq=24)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    decomposed = [trend, seasonal, residual]
    residual.dropna(inplace=True)
    return decomposed


def add_trend_and_seasonality_residual(decomposed, residual):
    """
    Adds the trend and seasonality to a residual dataset previously decomposed
    :param decomposed:
    :param residual:
    :return:
    """
    return decomposed[0] + decomposed[1] + residual


def scale_minmax_range(data, min_val, max_val, min_scale, max_scale):
    """
    Scales a dataset using a minmax function with range
    :param data:
    :param min_val:
    :param max_val:
    :param min_scale:
    :param max_scale:
    :return:
    """
    return ((data - min_val) / (max_val - min_val)) * (max_scale - min_scale) + min_scale


def rescale_minmax_range(data, min_val, max_val, min_scale, max_scale):
    """
    Rescales back a dataset using a minmax function with range
    :param data:
    :param min_val:
    :param max_val:
    :param min_scale:
    :param max_scale:
    :return:
    """
    return (data - min_scale) / (max_scale - min_scale) * (max_val - min_val) + min_val


def difference_timeseries(data, interval=1):
    """
    Creates a differenced dataset
    :param data:
    :param interval:
    :return:
    """
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return diff


def get_rmse(test, predictions):
    """
    Gets the RMSE by comparing the test data and predictions
    :param test:
    :param predictions:
    :return:
    """
    return np.sqrt(((np.asanyarray(test) - np.asanyarray(predictions)) ** 2).mean())


# invert differenced value
def inverse_difference_timeseries(data, predicted_value):
    """
    Gets the inverse dataset that was previously differenced
    :param data:
    :param predicted_value:
    :return:
    """
    return predicted_value + data[-1]


# ===============================
# ARMA
# ===============================

def evaluate_models_arma(data, p_values, q_values, is_stationary):
    """
    Evaluate combinations of p and q values for an ARMA model
    :param data:
    :param p_values:
    :param q_values:
    :param is_stationary:
    :return:
    """
    data = data.astype('float64')
    best_rmse, best_cfg = float("inf"), None
    for p in p_values:
        for q in q_values:
            try:
                if p == 0 and q == 0:
                    continue
                rmse = evaluate_model_arma(data, (p, q), is_stationary)
                if rmse < best_rmse:
                    best_rmse, best_cfg = rmse, (p, q)
            except:
                continue
    print('Best ARMA%s  RMSE=%.3f' % (best_cfg, best_rmse))
    return best_cfg


def evaluate_model_arma(data, order, is_stationary, verbose=1):
    """
    Evaluate an ARMA model for a given order (p,q) and return RMSE
    :param X:
    :param arma_order:
    :param is_stationary:
    :param verbose:
    :return:
    """
    start = time.clock()  # Start Timer
    # prepare training dataset
    data = data.astype('float64')
    train_size = int(len(data) * 0.90)
    train, test = data[0:train_size], data[train_size:]
    train_data = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        stationary_data = train_data
        # decomposed = remove_trend_and_seasonality(train)
        # If it's not stationary, difference or decompose first
        if not is_stationary:
            stationary_data = difference_timeseries(train_data, interval=1)
            # stationary_data = decomposed[2]

        model = ARMA(stationary_data, order=order)
        model_fit = model.fit(disp=0)
        predicted_value = model_fit.forecast()[0][0]
        # inverse the difference if not stationary
        if not is_stationary:
            predicted_value = inverse_difference_timeseries(train_data, predicted_value)
            # predicted_value = add_trend_and_seasonality_residual(decomposed, predicted_value)

        # print('inverse', predicted_value)

        predictions.append(predicted_value)
        train_data.append(test[t])

    end = time.clock() - start  # End Timer

    # print(predictions)
    # calculate RMSE
    rmse = get_rmse(predictions, test)
    print('ARMA%s RMSE:%.3f Time_Taken:%.3f' % (order, rmse, end))

    if verbose == 1:
        print(predictions)

    return rmse


# ===============================
# ARIMA
# ===============================

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models_arima(data, p_values, d_values, q_values, is_stationary):
    """
    Evaluate combinations of p, d and q values for an ARIMA model
    :param data:
    :param p_values:
    :param d_values:
    :param q_values:
    :param is_stationary:
    :return:
    """
    data = data.astype('float64')
    best_rmse, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    if p == 0 and q == 0:
                        continue
                    rmse = evaluate_model_arima(data, (p, d, q), is_stationary)
                    if rmse < best_rmse:
                        best_rmse, best_cfg = rmse, (p, d, q)
                except:
                    continue
    print('Best ARIMA%s  RMSE=%.3f' % (best_cfg, best_rmse))
    return best_cfg


def evaluate_model_arima(data, order, is_stationary, verbose=1):
    """
    Evaluate an ARIMA model for a given order (p,d,q) and return RMSE
    :param X:
    :param arma_order:
    :param is_stationary:
    :param verbose:
    :return:
    """
    start = time.clock()  # Start Timer
    # prepare training dataset
    data = data.astype('float64')
    train_size = int(len(data) * 0.90)
    train, test = data[0:train_size], data[train_size:]
    train_data = [x for x in train]
    # make predictions
    model = None
    predictions = list()
    for t in range(len(test)):
        stationary_data = train_data
        # decomposed = remove_trend_and_seasonality(train)
        # If it's not stationary, difference or decompose first
        if not is_stationary:
            stationary_data = difference_timeseries(train_data, interval=1)
            # stationary_data = decomposed[2]

        model = ARIMA(stationary_data, order=order)
        model_fit = model.fit(disp=0)
        predicted_value = model_fit.forecast()[0][0]
        # print('predicted_value1', model_fit.forecast()[0])
        # print('predicted_value2', predicted_value)
        # inverse the difference if not stationary
        if not is_stationary:
            predicted_value = inverse_difference_timeseries(train_data, predicted_value)
            # predicted_value = add_trend_and_seasonality_residual(decomposed, predicted_value)

        predictions.append(predicted_value)
        train_data.append(test[t])

    end = time.clock() - start  # End Timer
    # print(predictions)
    # calculate RMSE
    rmse = get_rmse(predictions, test)
    print('ARIMA%s RMSE:%.3f  Time_Taken:%.3f' % (order, rmse, end))

    if verbose == 1:
        print(predictions)

    return rmse


# ===============================
# Main
# ===============================

def main():
    """
    Main execution
    :return:
    """
    datasets = ['data/daily-births.csv', 'data/tesco-stock.csv', 'data/weather-madrid.csv']

    # load dataset
    print('#-----------------------------------------------')
    print('  ', datasets[0])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[0], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[0]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[1])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[1], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[0]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, q_values, False)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, False)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, False, 1)
    evaluate_model_arima(dataset, best_arima, False, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[2])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[2], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[0]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)


if __name__ == "__main__":
    main()
