# encoding:UTF-8
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # input data shape
    N, C, H, W = input_data.shape
    # calculate convolution rows and columns
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # padding zero to data set
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    # convolution result
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # scanning img matrix
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # transpose axis:N,OUT_H,OUT_W,C(channel),H,W then reshape matrix(N,C`)
    # C` = C * F_H*F_W
    # convolution needs each row for calculate result.
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1) # N*OH*OW,C`
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # col shape:N*OUT_H*OUT_W, C*FH*FW
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
