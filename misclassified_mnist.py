import numpy as np
from deep_convnet import DeepConvNet
import matplotlib.pyplot as plt
from dataset import load_mnist

(x_trian, t_trian), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()

network.load_params(file_name="./weight/deepnet.pkl")

print("caculating test accuracy ...")

classified_ids = []

acc = 0.
batch_size = 100

for i in range(x_test.shape[0] // batch_size):
    tx = x_test[i * batch_size:(i + 1) * batch_size]
    tt = t_test[i * batch_size:(i + 1) * batch_size]
    y = network.predict(tx)
    y = np.argmax(y, axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)

acc = acc / x_test.shape[0]
print(f"test accuracy is {acc}")

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()

max_view = 20
current_view = 1
fig = plt.figure()

mis_piars = {}

for i, val in enumerate(classified_ids == t_test):
    if not val:
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28),
                  cmap=plt.cm.gray_r,
                  interpolation="nearest")
        mis_piars[current_view] = (t_test[i], classified_ids[i])

        current_view += 1

        if current_view > max_view:
            break

print("=============misclassified result===============")
print("view index: (label, inference), ...")
print(mis_piars)
plt.show()
