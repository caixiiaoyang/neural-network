import numpy as np
import matplotlib.pyplot as plt
from dataset import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
network = DeepConvNet()
trainer = Trainer(network,
                  x_train,
                  t_train,
                  x_test,
                  t_test,
                  epochs=5,
                  mini_batch_size=100,
                  optimizer="adam",
                  optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

network.save_params(file_name="./weight/deepnet.pkl")
print("Save Network Parameters!")

plt.figure()
x = np.arange(len(trainer.train_acc_list))
plt.plot(x, trainer.train_acc_list, label="train_acc")
plt.plot(x, trainer.test_acc_list, label="test_acc")
plt.xlabel("step")
plt.ylabel("acc")
plt.legend()
plt.show()
