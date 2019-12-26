# encoding:UTF-8
# sigmoid = 激活函数
# h(x) = 1/(1+e^-x)
# soft max = 分类概率函数
# y(k)  = exp(ak)/(sum(exp(ai)))
# cross entropy error function
# - sum(tk * log yk)
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

# import all data of train
mnist = tf.keras.datasets.mnist
fashion_data = tf.keras.datasets.fashion_mnist


def sigmod(x):
    return 1 / (1 + np.exp(-x))  # exp(x) = e^x


def softmax(x):
    orig_shape = x.shape
    #n*m n = example number m=output arguments
    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)  # sum((Yi - t)**2)


def no_hot_cross_entropy_error(y: np.array, t: np.array) -> np.array:
    if y.ndim == 1:  # reshape shape to 1*n if shape is one dimension.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # row number is the total number of samples
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def hot_cross_entropy_error(y, t):
    if y.ndim == 1:  # reshape shape to 1*n if shape is one dimension.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # rows
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
