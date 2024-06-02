from deep_convnet import DeepConvNet
from dataset import load_mnist
import numpy as np

(x_trian, t_trian), (x_test, t_test) = load_mnist(flatten=False)
network = DeepConvNet()

network.load_params("./weight/deepnet.pkl")

# sampled = 10000
# x_test = x_test[:sampled]
# t_test = t_test[:sampled]

print("caculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))

x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)
print("caculate accuracy (float16) ...")
print(network.accuracy(x_test, t_test))
