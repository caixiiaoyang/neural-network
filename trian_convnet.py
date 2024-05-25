from simple_convnet import SimpleConvNet
from dataset import load_mnist
from common.trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt

(x_trian, t_trian), (x_test, t_test) = load_mnist(flatten=False)

convnet = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={
                            'filter_num': 30,
                            'filter_size': 5,
                            'pad': 0,
                            'stride': 1
                        },
                        hidden_size=100,
                        output_size=10,
                        weight_init_std=0.01)

trianer = Trainer(network=convnet,
                  x_train=x_trian,
                  t_train=t_trian,
                  x_test=x_test,
                  t_test=t_test,
                  epochs=20,
                  mini_batch_size=100,
                  optimizer="Adam",
                  optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trianer.train()

convnet.save_params("./weight/convnet.pkl")
print("saved net params")

plt.figure()
x = np.arange(20)
plt.plot(x, trianer.train_acc_list, label="trian")
plt.plot(x, trianer.test_acc_list, label="test")
plt.xlabel('epoch')
plt.ylabel('acc')
plt.ylim([0., 1.])
plt.legend()
plt.show()
