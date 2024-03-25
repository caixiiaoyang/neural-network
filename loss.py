import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def corss_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    loss = -np.sum(t * np.log(y + 1e-7)) / batch_size
    return loss


def cross_entropy_error1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)

    batch_size = y.shape[0]
    loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return loss


if __name__ == "__main__":
    t = np.array([1, 2, 2])
    y = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5], [0.2, 0.1, 0.7]])
    a = y[np.arange(3), t]
    print([np.arange(3), t])
