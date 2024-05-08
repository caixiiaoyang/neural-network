import numpy as np
import matplotlib.pyplot as plt
from dataset import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  one_hot_lable=True)

x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def _train(lr, weight_decay, epoch=50):
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            weight_decay_lambda=weight_decay)
    trainer = Trainer(network,
                      x_train,
                      t_train,
                      x_val,
                      t_val,
                      epochs=epoch,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': lr},
                      verbose=False)

    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


optimization_trial = 100
results_val = {}
result_trian = {}

for _ in range(optimization_trial):
    weight_decay = 10**np.random.uniform(-8, -4)
    lr = 10**np.random.uniform(-6, -2)

    val_acc_list, train_acc_list = _train(lr, weight_decay)
    print(f'val acc {val_acc_list[-1]}|lr:{lr}, weight decay:{weight_decay}')
    key = f"lr:{lr}, weight decay: {weight_decay}"

    results_val[key] = val_acc_list
    result_trian[key] = train_acc_list

print("=============Hyper-Parameter Optimization Result ==================")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(),
                                key=lambda x: x[1][-1],
                                reverse=True):
    print(f'Best-{i+1}(val acc:{val_acc_list[-1]})|{key}')
    plt.subplot(row_num, col_num, i + 1)
    plt.title(f'Best-{i+1}',
              fontsize='x-small',
              verticalalignment='top',
              horizontalalignment='center')
    if i % 5:
        plt.yticks([])
    plt.ylim(0., 1.)
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, result_trian[key], linestyle='--')
    i += 1
    if i >= graph_draw_num:
        break

plt.show()
