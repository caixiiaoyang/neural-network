from common.layer import BatchNormalization
import numpy as np
from dataset import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend

(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  one_hot_lable=True)

network = MultiLayerNetExtend(input_size=784,
                              hidden_size_list=[100, 100],
                              output_size=10,
                              use_batchnorm=True)
x_batch = x_train[:100]
t_batch = t_train[:100]

grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_grad(x_batch, t_batch)

for key in grad_backprop.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"key:{diff}")