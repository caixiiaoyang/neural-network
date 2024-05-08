import numpy as np


def shuffle_dataset(x, t):
    x_shape = x.shape
    t_shape = t.shape
    x = x.reshape(x_shape[0], -1)
    t = t.reshape(t_shape[0], -1)
    permutation = np.random.permutation(x_shape[0])

    x = x[permutation, :]
    t = t[permutation, :]
    x = x.reshape(x_shape)
    t = t.reshape(t_shape)

    return x, t
