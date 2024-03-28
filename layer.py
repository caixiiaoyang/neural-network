import numpy as np


def softmax1d(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def softmax(a):
    if a.ndim == 1:
        return softmax1d(a)

    c = np.max(a, axis=1)
    c = c.reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1)
    sum_exp_a = sum_exp_a.reshape(-1, 1)

    return exp_a / sum_exp_a
