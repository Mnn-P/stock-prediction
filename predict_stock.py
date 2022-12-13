import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pandas_datareader import data, wb
from tensorflow.contrib import rnn

sequence = 7
inputD = 5
outD = 1


def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def dataConvert(data):
    data = data[::-1]
    data = MinMaxScaler(data)
    x = data
    y = data[:, [-1]]
    x_data = []
    y_data = []
    for i in range(0, len(x) - sequence):
        _x = x[i:i + sequence]
        _y = y[i + sequence]

        x_data.append(_x)
        y_data.append(_y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data