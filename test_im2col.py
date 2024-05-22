from common.util import im2col, col2im
import numpy as np

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, 1, 0)
im1 = col2im(col1, (1, 3, 7, 7), 5, 5, 1, 0)
print(x1[0][1][2][3])
print(im1[0][1][2][3])
