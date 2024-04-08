import numpy as np
from common.layer import *
from collections import OrderedDict
from grad import numerical_gradient


class TwoLayerNet:

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01) -> None:
        #初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(
            hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        #生成层
        self.layer = OrderedDict()
        self.layer["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layer["Relu1"] = Relu()
        self.layer["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layer.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / x.shape[0]

        return acc

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}

        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layer.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layer["Affine1"].dW
        grads["b1"] = self.layer["Affine1"].db
        grads["W2"] = self.layer["Affine2"].dW
        grads["b2"] = self.layer["Affine2"].db

        return grads
