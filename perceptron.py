# a simple implementation
import numpy as np


def AND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp >= theta:
        return 1
    else:
        return 0


def OR1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.4
    tmp = w1 * x1 + w2 * x2
    if tmp >= theta:
        return 1
    else:
        return 0


def NAND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp >= theta:
        return 0
    else:
        return 1


# thr implementation use weight and bias
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.7
    tmp = np.sum(w * x) + theta
    if tmp >= 0:
        return 1
    else:
        return 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.4
    tmp = np.sum(w * x) + theta
    if tmp >= 0:
        return 1
    else:
        return 0


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.7
    tmp = np.sum(w * x) + theta
    if tmp >= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == "__main__":
    invars = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    for invar in invars:
        x1, x2 = invar
        out_and1 = AND1(x1, x2)
        out_and = AND(x1, x2)
        out_or1 = OR1(x1, x2)
        out_or = OR(x1, x2)
        out_nand1 = NAND1(x1, x2)
        out_nand = NAND(x1, x2)
        out_xor = XOR(x1, x2)
        print(
            "****************************************************************")
        print(f"the input is {x1,x2}")
        print(f"tht output of AND and AND1 is {out_and, out_and1}")
        print(f"the output of OR and OR1 is {out_or, out_or1}")
        print(f"the output of out_nand and out_nand1 is {out_nand, out_nand1}")
        print(f"the output of out_xor is {out_xor}")
