import numpy as np


def step_fun(x):
    if x > 0:
        return 1
    else:
        return 0


def step(x):
    x = np.array(x)
    y = x > 0
    return y.astype(np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def identity(x):
    return x


if __name__ == "__main__":
    x1 = np.array([-1.0, 1.0, 2.0])
    x2 = 3.0
    print(step_fun(x2), step(x1), step(x2))
