import numpy as np
from loss import corss_entropy_error
from layer import softmax
from grad import numerical_gradient


class simpleNet:

    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        losses = corss_entropy_error(y, t)

        return losses


if __name__ == "__main__":
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    loss = net.loss(x, t)
    print(loss)
    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(dW)
