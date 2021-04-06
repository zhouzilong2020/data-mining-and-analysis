import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import *


def load_dataset(data_path, sample_number=100, exclude_dim=[0]):
    """Load dataset from a .mat file.
       Args:
            :param exclude_dim:
            :param data_path:
            :param sample_number:
    """
    if sample_number is not np.inf:
        data = np.loadtxt(data_path, delimiter=',')[:sample_number, :]
    else:
        data = np.loadtxt(data_path, delimiter=',')
    ndim = [i for i in range(data.shape[1])]
    return data[:, [False if i in exclude_dim else True for i in ndim]]


def normalize(xs):
    """Apply normalization to given data xs

       Args:
            xs: An numpy array

       Returns:
           z_xs: normalized result
    """
    return (xs - np.min(xs, axis=0)) / (np.max(xs, axis=0) - np.min(xs, axis=0))


def plot(save_path, data, type_label, title, x_label, y_label):
    plt.figure()
    rcParams['axes.unicode_minus'] = False
    rcParams['font.sans-serif'] = ['Simhei']

    temp = dict()
    for i in range(data.shape[0]):
        temp[f"{type_label[i]}"] = data[i]

    label = pd.DataFrame(temp)

    df = pd.DataFrame(label)
    df.plot.box(title=title)

    plt.xlabel(xlabel=x_label, fontsize=16)
    plt.ylabel(ylabel=y_label, fontsize=16)
    plt.grid(linestyle="--", alpha=0.8)
    # print(df.describe())  # 显示中位数、上下四分位数、标准偏差等内容
    plt.savefig(save_path)
