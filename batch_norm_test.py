import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD
from dataset import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  one_hot_lable=True)
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
trian_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def _trian(weight_init_std):
    bn_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784,
                                  hidden_size_list=[100, 100, 100, 100, 100],
                                  output_size=10,
                                  weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    trian_acc_list = []
    bn_trian_acc_list = []

    iter_per_epoch = max(trian_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(100000000):
        batch_mask = np.random.choice(trian_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            trian_acc_list.append(train_acc)
            bn_trian_acc_list.append(bn_train_acc)

            print(
                f"epoch:{epoch_cnt}|trian_acc:{train_acc},bn_trian_acc:{bn_train_acc}"
            )
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
    return trian_acc_list, bn_trian_acc_list


weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print(f"==============={i+1}/16==============")
    train_acc_list, bn_trian_acc_list = _trian(w)
    plt.subplot(4, 4, i + 1)
    plt.title("W:" + str(w)[:4])
    if i == 15:
        plt.plot(x,
                 bn_trian_acc_list,
                 label='Batch Normalization',
                 markevery=2)
        plt.plot(x,
                 train_acc_list,
                 linestyle='--',
                 label='Normal(without BatchNorm)',
                 markevery=2)
    else:
        plt.plot(x, bn_trian_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)
    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")

    plt.legend(loc='lower right')
plt.show()
