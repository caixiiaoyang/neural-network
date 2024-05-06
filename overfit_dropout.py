import numpy as np
import matplotlib.pyplot as plt
from dataset import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                  one_hot_lable=True)

x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = True
dropout_ratio = 0.2

network = MultiLayerNetExtend(input_size=784,
                              hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10,
                              use_dropout=use_dropout,
                              dropout_ration=dropout_ratio)

trainer = Trainer(network,
                  x_train,
                  t_train,
                  x_test,
                  t_test,
                  epochs=301,
                  mini_batch_size=100,
                  optimizer='sgd',
                  optimizer_param={'lr': 0.01},
                  verbose=True)

trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

markers = {'train': 'o', 'test': 's'}

x = np.arange(len(train_acc_list))

plt.figure()
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0,1.0)
plt.legend(loc='lower right')
plt.show()