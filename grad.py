import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tem_val = x[idx]

        x[idx] = tem_val + h
        fxh1 = f(x)
        x[idx] = tem_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tem_val

    return grad


def numerical_gradient2d(f, x):
    grads = np.zeros_like(x)
    if x.ndim == 1:
        return numerical_gradient1d(f, x)
    for idx in range(len(x)):
        grad = numerical_gradient1d(f, x[idx])
        grads[idx] = grad
    return grads


def numerical_gradient(f, x):
    grads = np.zeros_like(x)
    if x.ndim == 1 or x.ndim == 2:
        return numerical_gradient2d(f, x)
    for idx in range(len(x)):
        grad = numerical_gradient2d(f, x[idx])
        grads[idx] = grad
    return grads


def gradient_descent(f, init_x, lr=1e-2, step=100):
    x = init_x

    for i in range(step):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
