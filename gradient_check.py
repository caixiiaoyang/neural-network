import numpy as np
from dataset import load_mnist
from twolayernet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  one_hot_lable=True)
network = TwoLayerNet(784, 50, 10)
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+": "+str(diff))
