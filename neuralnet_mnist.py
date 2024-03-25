from dataset import load_mnist
import os
import numpy as np
import pickle
from activate import sigmoid
from layer import softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normolize=True,
                                                      flatten=True,
                                                      one_hot_lable=False)
    return x_test, t_test


def init_network():
    if not os.path.exists("./weight/"):
        os.makedirs("./weight/")
    if not os.path.exists("./weight/sample_weight.pkl"):
        layer_size = [784, 50, 100, 10]
        network = {}
        weights = list(zip(layer_size[:-1], layer_size[1:]))
        bias = layer_size[1:]

        for i, (w, b) in enumerate(zip(weights, bias)):
            network[f"W{i}"] = np.random.normal(size=w)
            network[f"b{i}"] = np.random.normal(size=(b, ))
        with open("./weight/sample_weight.pkl", "wb") as f:
            pickle.dump(network, f)
    with open("./weight/sample_weight.pkl", "rb") as f:
        network_weight = pickle.load(f)

    return network_weight


def predict(network, x):
    W1, W2, W3 = network["W0"], network["W1"], network["W2"]
    b1, b2, b3 = network["b0"], network["b1"], network["b2"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print(f"Accuracy:{accuracy_cnt/len(x)}")
    """进行批处理"""

    batch_size = 100
    acc_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        t_batch = t[i:i + batch_size]
        y_batch = predict(network, x_batch)
        acc_cnt += np.sum(t_batch == np.argmax(y_batch, axis=1))

    print(f"Batch acc is {acc_cnt/len(x)}")
