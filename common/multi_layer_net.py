import sys, os

sys.path.append(os.pardir)
import numpy as np
from common.layer import Sigmod, Affine, Relu, SoftmaxWithLoss
from grad import numerical_gradient
from collections import OrderedDict


class MultiLayerNet:

    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation="relu",
                 weight_init_std="relu",
                 weight_decay_lambda=0) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self._init_weight(weight_init_std)

        activation_layer = {'sigmod': Sigmod, 'relu': Relu}

        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers["Activation_function" +
                        str(idx)] = activation_layer[activation]()
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def _init_weight(self, weight_init_std):
        all_size_list = [self.input_size
                         ] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmod', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])

        return acc

    def numerical_grad(self, x, t):
        lossW = lambda W: self.loss(x, t)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(
                lossW, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(
                lossW, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1

        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(
                idx)].dW + self.weight_decay_lambda * self.layers['Affine' +
                                                           str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
