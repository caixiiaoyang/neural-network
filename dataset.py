# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError("You should use Python 3.x")
import os.path
import gzip
import pickle
import os
import numpy as np
from PIL import Image

url_base = 'http://yann.lecun.com/exdb/mnist/'
ket_file = {
    "train_image": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_image": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}
data_path = "./data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

save_file = data_path + "mnist.pkl"
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path):
        return

    print("Downing " + file_name + "...")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done!")


def down_mnist():
    for v in ket_file.values():
        _download(v)


def _load_label(file_name):
    file_path = os.path.join(data_path, file_name)
    print("Coverting " + file_name + "to numpy arrry ...")
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done !")
    return labels


def _load_image(file_name):
    file_path = os.path.join(data_path, file_name)

    print("Converting " + file_name + "to numpy array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, img_size)
    print("Done !")
    return data


def convert_numpy():
    dataset = {}
    dataset["train_image"] = _load_image(ket_file["train_image"])
    dataset["test_image"] = _load_image(ket_file["test_image"])
    dataset["train_label"] = _load_label(ket_file["train_label"])
    dataset["test_label"] = _load_label(ket_file["test_label"])

    return dataset


def init_mnist():
    down_mnist()
    dataset = convert_numpy()
    print("Creating pickle file ....")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done !")


def _change_one_hot_label(X, numclass=10):
    return np.eye(numclass)[X]


def img_show(img):
    plt_img = Image.fromarray(img)
    plt_img.show()


def load_mnist(normolize=True, flatten=True, one_hot_lable=False):
    """load the dataset of mnist

    Args:
        normolize (bool, optional):  Defaults to True.
        flatten (bool, optional):  Defaults to True.
        one_hot_lable (bool, optional):  Defaults to False.

    Returns:
        dict: the dataset
    """
    init_mnist()

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    if normolize:
        dataset["train_image"] = dataset["train_image"].astype(
            np.float32) / 255.0
        dataset["test_image"] = dataset["test_image"].astype(
            np.float32) / 255.0

    if one_hot_lable:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ["train_image", "test_image"]:
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_image"],
            dataset["train_label"]), (dataset["test_image"],
                                      dataset["test_label"])


if __name__ == "__main__":
    (train_img, train_label), (test_img,
                               test_label) = load_mnist(normolize=False,
                                                        flatten=True,
                                                        one_hot_lable=False)
    img = train_img[0]
    label = train_label[0]
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    print(label)
    img_show(img)
