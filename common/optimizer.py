import numpy as np


class SGD:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class SGDM:
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) +
                                                       1e-7)


class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99) -> None:
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

            for key in params.keys():
                self.h[key] *= self.decay_rate
                self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) +
                                                       1e-7)


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iters = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iters += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.iters) / (
            1 - self.beta1**self.iters)

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 -
                                                      self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key]**2)
            params[key] -= lr_t * self.m[key] / np.sqrt(self.v[key] + 1e-7)
