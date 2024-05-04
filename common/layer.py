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


class BatchNormalization:

    def __init__(self,
                 gamma,
                 beta,
                 momentum=0.9,
                 running_mean=None,
                 running_var=None) -> None:
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N = x.shape[0]
            x = x.reshape(N, -1)
        out = self._forward(x, train_flg)
        out = out.reshape(*self.input_shape)
        return out

    def _forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-6)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (
                1 - self.running_mean) * mu
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 1e-6))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self._backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def _backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(dout * self.xn, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = -np.sum(dxc, axis=0)
        dx = dxc + dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx
