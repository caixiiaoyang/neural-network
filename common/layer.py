import numpy as np

class Relu:

    def __init__(self) -> None:
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0

        dx = dout
        return dx


class Sigmod:

    def __init__(self) -> None:
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        self.out = out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)

        return dx


class Affine:

    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


def softmax1d(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def softmax(a):
    if a.ndim == 1:
        return softmax1d(a)

    c = np.max(a, axis=1)
    c = c.reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1)
    sum_exp_a = sum_exp_a.reshape(-1, 1)

    return exp_a / sum_exp_a


def corss_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    loss = -np.sum(t * np.log(y + 1e-7)) / batch_size
    return loss


class SoftmaxWithLoss:

    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = corss_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
