import numpy as np
from dataset import load_mnist
from twolayernet import TwoLayerNet
import matplotlib.pyplot as plt
from tqdm import tqdm
(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  one_hot_lable=True)
network = TwoLayerNet(784, 100, 10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []

iter_per_epoch = max(train_size / batch_size, 1)
with tqdm(desc="train", total=iters_num) as t:
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in network.params:
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        t.set_postfix(loss=loss)
        t.update(1)
        train_loss_list.append(loss)

x = np.arange(0, iters_num)
y = np.array(train_loss_list)
plt.figure()
plt.plot(x, y)
plt.show()
