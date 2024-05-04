from common.layer import BatchNormalization
import numpy as np
from grad import numerical_grad

inputs = np.random.randn(8, 16)
gamma = np.ones(16, )
beta = np.zeros(16, )

bn = BatchNormalization(gamma, beta)
out = bn.forward(inputs)
dout = np.ones_like(out)
bn_f = lambda x: bn.forward(x)
backward_dx = bn.backward(dout)
numerical_dx = numerical_grad(bn_f, inputs)
print(np.allclose(backward_dx, numerical_dx))
