import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet


def filter_show(filters, nx=8, margin=3, scale=10):
    FN, C, FH, FW = filters.shape

    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0,
                        right=1,
                        bottom=0,
                        top=1,
                        hspace=0.05,
                        wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


network = SimpleConvNet()
filter_show(network.params['W1'])

network.load_params('E:/network-learn/neural-network/weight/convnet.pkl')

filter_show(network.params['W1'])
