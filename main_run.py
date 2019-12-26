from collections import OrderedDict

from nerve_frame.back_ward_differential import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        weight_init_std = np.sqrt(2) / (np.sqrt(input_size))
        # initialized params
        self.parameter = {}
        # input nerve hierarchy
        self.parameter["W1"] = weight_init_std * \
                               np.random.randn(input_size, hidden_size)
        self.parameter["b1"] = np.zeros(hidden_size)
        # second hierarchy nerve network
        weight_init_std = np.sqrt(2) / (np.sqrt(hidden_size))
        self.parameter["W2"] = weight_init_std * \
                               np.random.randn(hidden_size, output_size)
        self.parameter["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.parameter["W1"], self.parameter["W2"]
        b1, b2 = self.parameter["b1"], self.parameter["b2"]
        # calculate network
        a1 = np.dot(x, W1) + b1
        z1 = sigmod(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return no_hot_cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_w, self.parameter["W1"])
        grads["b1"] = numerical_gradient(loss_w, self.parameter["b1"])
        grads["W2"] = numerical_gradient(loss_w, self.parameter["W2"])
        grads["b2"] = numerical_gradient(loss_w, self.parameter["b2"])
        return grads


class NewTwoLayerNet(TwoLayerNet):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, output_size, weight_init_std=0.01):
        super(NewTwoLayerNet, self).__init__(input_size, first_hidden_size, second_hidden_size, weight_init_std)
        self.parameter["W3"] = np.random.randn(second_hidden_size, output_size) * (
                np.sqrt(2) / np.sqrt(second_hidden_size))
        self.parameter["b3"] = np.zeros(output_size)
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.parameter["W1"], self.parameter["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.parameter["W2"], self.parameter["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine3"] = Affine(self.parameter["W3"], self.parameter["b3"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)  # calculate nerve network
        return self.lastLayer.forward(y, t)  # cross_entropy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)  # the parameters forward propagate to nerve network that will record them
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # setting hierarchy
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        return grads


class SimpleConvertNet(NewTwoLayerNet):
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=50, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # rows and columns of output shape = square matrix
        first_conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = filter_num * (first_conv_output_size  /  2 ) ** 2
        # first_pool_output_size = first_conv_output_size / 2
        # second_conv_output_size = (first_pool_output_size - 4) + 1
        # pool_output_size = int(filter_num * (second_conv_output_size / 2) ** 2)

        # 初始化权重
        self.params = {}
        # first layer output N*C*H*W shape.
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # pool shape is = N*C*P_H*P_W
        # self.params["W2"] = np.random.randn(filter_num, input_dim[0], 4, 4) * weight_init_std
        # self.params["b2"] = np.zeros(filter_num)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(int(pool_output_size), hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        self.parameter = self.params
        # 生成层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        # self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Pool1'] = Average_Pooling(pool_h=2, pool_w=2, stride=2)
        # self.layers["Conv2"] = Convolution(self.params["W2"], self.params["b2"])
        # self.layers["Relu2"] = Relu()
        # self.layers["Poo2"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        # grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


class DeepConvNet:
    """deep nerve network
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):
        # 初始化权重===========
        # 各层的神经元平均与前一层的几个神经元有连接（TODO:自动计算）
        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用ReLU的情况下推荐的初始值

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(
                [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W8'], self.params['b8']))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 17)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads


class SGD:
    def __init__(self, rate):
        self.rate = rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.rate * grads[key]


class Momentum(SGD):
    def __init__(self, rate, momentum=0.9):
        super(Momentum, self).__init__(rate)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.rate * grads[key]
            params[key] += self.v[key]


class AdaGrad(SGD):
    def __init__(self, lr=0.01):
        super(AdaGrad, self).__init__(lr)
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
        for key, val in params.items():
            self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= (self.rate * grads[key] / (np.sqrt(self.h[key]) + 1e-7))


class Nesterov(SGD):
    def __init__(self, lr=0.01, momentum=0.9):
        super(Nesterov, self).__init__(lr)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.rate * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.rate * grads[key]


class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


if __name__ == '__main__':
    # (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = fashion_data.load_data()
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = mnist.load_data()
    # regularization data
    fashion_x_train, fashion_x_test = fashion_x_train / 255.0, fashion_x_test / 255.0
    # fashion_x_train, fashion_x_test = fashion_x_train.reshape(60000, 784), fashion_x_test.reshape(10000, 784)
    learning_rate = 0.1
    sgd = SGD(learning_rate)
    momentum = Momentum(learning_rate)
    nesterov = Nesterov(learning_rate)
    ada = AdaGrad(learning_rate)
    adam = Adam()
    # two_net = NewTwoLayerNet(784, 50, 50, 10)
    cnn_network = SimpleConvertNet()
    deep_network =DeepConvNet()
    # single channel image
    fashion_x_train = fashion_x_train.reshape(60000, 1, 28, 28)
    fashion_x_test = fashion_x_test.reshape(10000, 1, 28, 28)
    # batch_mask = np.random.choice(60000,3)
    # x_batch = fashion_x_train[batch_mask]
    # y_batch = fashion_y_train[batch_mask]
    # grad_numerical = two_net.numerical_gradient(x_batch,y_batch)
    # grad_backprop = two_net.gradient(x_batch,y_batch)
    # print(grad_numerical
    # print(grad_backprop)
    #
    real_network = cnn_network
    s = 100 * real_network.accuracy(fashion_x_test[:], fashion_y_test[:])
    for i in range(5000):
        batch_mask = np.random.choice(60000, 100)
        x_batch = fashion_x_train[batch_mask]
        y_batch = fashion_y_train[batch_mask]
        # calculate gradient
        grads = real_network.gradient(x_batch, y_batch)
        #     grads = two_net.gradient(x_batch, y_batch)
        #     #     # update network parameter
        #     adam.update(two_net.parameter, grads)
        sgd.update(real_network.params, grads)
        #     loss = two_net.loss(x_batch, y_batch)
        loss = real_network.loss(x_batch, y_batch)
        print(loss, f"第{i}次学习")
    print(f"未训练识别成功率{s}%")
    # print(f"已训练识别成功率{100 * two_net.accuracy(fashion_x_test[:], fashion_y_test[:])}%")
    print(f"学习后识别成功率{100 * real_network.accuracy(fashion_x_test[:], fashion_y_test[:])}%")
