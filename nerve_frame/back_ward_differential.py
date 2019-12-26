# encoding:UTF-8
from nerve_frame.gradient_function import *
from utility.image_to_column import *


class MultiplicationLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# activation function
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0  # True if x <=0
        out = x.copy()
        out[self.mask] = 0  # elect values less than zero than assign zero to them
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmod:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # 1/（1+e^-x), ->=  e^-x/(1+e^-x)^2->=y(1-y): y=1/(1+e^x)
        dx = dout * self.out * (1.0 - self.out)
        return dx


# Affine matrix
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


# classify data by label then calculate loss
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # loss argument
        self.y = None  # output of the softmax
        self.t = None  # monitor of the data

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)  # The data input to softmax function
        self.loss = no_hot_cross_entropy_error(self.y, t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y[np.arange(batch_size), self.t] - 1)
        y = self.y.copy()
        y[np.arange(batch_size), self.t] = dx
        return y / batch_size



# nerve frame
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        """Convolution nerve Layer"""
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # middle data that  give to backward method using)
        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x):
        # All the kernel are same shape.
        # FN:number of all kernel,C:number of chanel,
        # FH:number of rows,FW:number of weight.
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape  # input data's shape
        out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
        out_w = int((W + 2 * self.pad - FW) / self.stride + 1)
        # convert input data to matrix
        col = im2col(x, FH, FW, self.stride, self.pad)
        # each column is kernel arguments
        # self.W = FN*C*FH*FW , now reshape: FN,C*FH*W .T: C*FH*FW
        col_w = self.W.reshape(FN, -1).T
        out = np.dot(col, col_w) + self.b  # col N*OH*OW,C` DOT C*FH*FW,FN
        # out data's shape is N,FN,OH,OW
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # save data that will calculate for back propagate method.
        self.x = x
        self.col = col
        self.col_W = col_w
        return out

    def backward(self, dout):
        """FN:filter number(be equal kernel number)
             C:channel number of each kernel
             FH: height of kernel
             FW: weight of kernel
        """
        FN, C, FH, FW = self.W.shape
        # 0:N,  1:FN,  2:OUT_H, 3:OUT_W transpose axis to 0:N,  2:OUT_H,  3:OUT_W, 1:FN
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)  # dout shape:(N*OUT_H*OUT_W,FN)
        # sum each kernel's rows that shape is 1*FN
        # attention!!!: C` != C that C` is : channel number * out_h * out_w
        self.db = np.sum(dout, axis=0)  # 1 * FN
        # col shape:(N*OUT_H*OUT_W,C').T=(C',N*OUT_H*OUT_W)
        self.dW = np.dot(self.col.T, dout)  # shape: C` * FN
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)  # col_W=(C * FH*FW,FN).T = reverse axis
        # dcol matrix shape = N*OUT_H*OUT_W,C*FH*FW
        # so it needs reshape like to  input data shape
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

# pooling layer
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        """pooling matrix,extract maximum number in the certain range of matrix"""
        N, C, H, W = x.shape
        out_h = int((H - self.pool_h) / self.stride + 1)
        out_w = int((W - self.pool_w) / self.stride + 1)

        # Unfold matrix
        # col shape:N*OH*OW, C*FH*FW
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # convert col matrix to square matrix
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # maximum numbers
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        # save maximum index
        arg_max = np.argmax(col, axis=1)
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        # n c h w -> n h w c
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        # rows = sum of all dout's element , columns = sum of pool elements
        dmax = np.zeros((dout.size, pool_size))
        # Assign number to maximum column index place
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()  # flatten: all elements unfold to one row
        # dout shape: N,C,OUT_H,OUT_W,now dmax shape = N,OUT_H,OUT_W,C ,POOL_SIZE
        dmax = dmax.reshape(dout.shape + (pool_size,))
        # convert to like dcol's matrix shape:N*OH*OW, C*FH*FW(FH*FW=POOL_SIZE)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class Average_Pooling(Pooling):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        # initialization data method keep same  as the parent method.
        super(Average_Pooling, self).__init__(pool_h, pool_w, stride, pad)
        self.average = None

    def forward(self, x):
        """pooling data"""
        N, C, H, W = x.shape
        out_h = int((H - self.pool_h) / self.stride + 1)
        out_w = int((W - self.pool_w) / self.stride + 1)

        # Unfold matrix
        # col shape:N*OH*OW, C*FH*FW
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # convert col matrix to square matrix
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # maximum numbers
        out = np.average(col, axis=1)
        self.average = out.copy()
        self.x = x
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        # n c h w -> n h w c
        dout /= (self.pool_w * self.pool_h)
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_w * self.pool_h
        daverage = np.ones((dout.size, pool_size))
        daverage *= dout.reshape(-1,1)
        daverage = daverage.reshape(dout.shape + (pool_size,))
        # convert to like dcol's matrix shape:N*OH*OW, C*FH*FW(FH*FW=POOL_SIZE)
        dcol = daverage.reshape(daverage.shape[0] * daverage.shape[1] * daverage.shape[2], -1)
        # The inverse of this im2col function.
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx


if __name__ == '__main__':
    # functional test
    test_array = np.array([-1, 0, 2, -3.1])
    r = Relu()
    print(r.forward(test_array))
    print(r.backward(test_array))
