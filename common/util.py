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


def get_im2col_indices(x_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = x_shape
    assert (H + 2 * pad - filter_h) % stride == 0
    assert (W + 2 * pad - filter_w) % stride == 0
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    i0 = np.repeat(np.arange(filter_h), filter_w)
    i0 = np.tile(i0, C)
    j0 = np.tile(np.arange(filter_w), filter_h)
    j0 = np.tile(j0, C)
    i1 = stride * (np.repeat(np.arange(out_h), out_w))
    j1 = stride * (np.tile(np.arange(out_w), out_h))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), filter_h * filter_w).reshape(-1, 1)
    return (i, j, k)


def im2col_indices(input_data, filter_h, filter_w, stride=1, pad=0):
    p = pad
    x_paded = np.pad(input_data, [[0, 0], [0, 0], [p, p], [p, p]],
                     mode="constant")
    (i, j, k) = get_im2col_indices(input_data.shape, filter_h, filter_w,
                                   stride, pad)
    cols = x_paded[:, k, i, j]
    C = input_data.shape[1]
    cols = cols.transpose(0, 2, 1).reshape(-1, filter_h * filter_w * C)

    return cols


def col2im_indices(cols, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    h_pad, w_pad = H + 2 * pad, W + 2 * pad
    x_paded = np.zeros((N, C, h_pad, w_pad), dtype=cols.dtype)

    (i, j, k) = get_im2col_indices(input_shape, filter_h, filter_w, stride,
                                   pad)

    cols_reshaped = cols.reshape(N, -1,
                                 filter_h * filter_w * C).transpose(0, 2, 1)

    np.add.at(x_paded, (slice(None), k, i, j), cols_reshaped)

    if pad == 0:
        return x_paded
    else:
        return x_paded[:, :, pad:-pad, pad:-pad]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == "__main__":
    input_shape = (2, 2, 4, 4)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)

    print(x)
    print("***********************************************")
    col = im2col_indices(x, 3, 3, pad=0, stride=1)
    print(col)
    print("***********************************************")
    col = col.reshape(-1, 3 * 3)
    print(col)
    out = np.max(col, axis=1)
    out = out.reshape(2, 2, 2, 2).transpose(0, 3, 1, 2)
    print(out)
    # x = col2im_indices(col, input_shape, 3, 3, pad=0, stride=1)
    # print(x)
