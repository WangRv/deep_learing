# encoding:UTF-8
from utility.activate import *

# differential coefficient
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# matrix
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    for idx in it:
        ix = it.multi_index
        idx += h
        fxh1 = f(x)
        # f-h calculate
        idx -= 2 * h
        fxh2 = f(x)
        grad[ix] = (fxh1 - fxh2) / (2 * h)
        idx += h  # recovery data
    return grad

# gradient
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        # start training
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def matrix_gradient(f, x, lr=0.01, step_num=100):
    W = x
    for i in range(step_num):
        grad = numerical_gradient(f, W)
        W -= lr * grad
    return W

# gradient nerve network
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = hot_cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    net = SimpleNet()
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    t = np.array([0, 0, 1])  # Correct index of label
    f = lambda W: net.loss(x, t)  # loss function
    # test_result = numerical_gradient(f, net.W)
    # print(test_result)
    print(net.W)
    matrix_gradient(f, net.W, step_num=2000)
    print(net.W)
