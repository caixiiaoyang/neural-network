import numpy as np
from dataset import load_mnist
from two_layer_net import TwoLayerNet
from tqdm import tqdm
import matplotlib.pyplot as plt
(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  flatten=True,
                                                  one_hot_lable=True)
train_loss_list = []

iter_num = 1000
trian_size = x_train.shape[0]
batch_size = 10
learn_rate = 0.1
network = TwoLayerNet(784, 100, 10)
with tqdm(desc="train", total=iter_num) as t:
    for i in range(iter_num):
        batch_mask = np.random.choice(trian_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in network.params:
            network.params[key] -= learn_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        t.set_postfix(loss=loss)
        t.update(1)
        train_loss_list.append(loss)

    x = np.arange(0, iter_num)
    y = np.array(train_loss_list)
    plt.figure()
    plt.plot(x, y)
    plt.show()
