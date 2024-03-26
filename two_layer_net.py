import numpy as np
from grad import numerical_gradient
from layer import softmax
from loss import corss_entropy_error
from activate import sigmoid


class TwoLayerNet:

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01) -> None:
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(
            hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return corss_entropy_error(y, t)

    def gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


if __name__ == "__main__":
    net = TwoLayerNet(784, 100, 10)
    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)
    y = net.predict(x)
    loss = net.loss(x, t)
    grads = net.gradient(x, t)
    print(y)
    print(loss)
    print(grads)
