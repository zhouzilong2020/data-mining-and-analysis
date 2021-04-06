import utils
import numpy as np
import os


def comb(items, k):
    """ give the combination array of a given set of value

    Args:
         items: a list to be calculated its combination array
         k: the size of combination

    Returns:
        result: all the possible combination of the given list with size k
    """
    if k == 0 or len(items) == 0 or k > len(items):
        return []
    result = []
    with_i = comb(items[1:], k - 1)
    if len(with_i) > 0:
        for j in with_i:
            if type(j) is list:
                result.append([items[0]] + j)
            else:
                result.append([items[0], j])
    else:
        result.append(items[0])

    without_i = comb(items[1:], k)
    if len(without_i) > 0:
        for j in without_i:
            result.append(j)
    return result


def getDis(data):
    def e_distance(x_1, x_2):
        return np.sqrt(np.sum(np.power(x_1 - x_2, 2)))

    result = []
    len_data = len(data)
    for i in range(len_data):
        for j in range(i + 1, len_data):
            result.append(e_distance(data[i], data[j]))
    return np.array(result)


def getMean(dist):
    return dist.mean(axis=0)


def getDiff(dist):
    return (dist.max(axis=0) - dist.min(axis=0)) / dist.max(axis=0)


def main(out_dir, data_path, sample_number=100, exclude_dim=[0]):
    # prepare data
    data = utils.load_dataset(data_path, sample_number, exclude_dim)
    data = utils.normalize(data)
    ndim = [i for i in range(data.shape[1])]

    dist_mean = []
    dist_diff = []
    for i in range(data.shape[1] - 1):
        # get selected attributes(dimensions) to calculate distance
        prop_combs = comb(ndim, i + 1)
        dist_mean_temp = []
        dist_diff_temp = []
        for properties_comb in prop_combs[:data.shape[1] - 1]:
            dist = getDis(data[:, properties_comb])
            # get mean distance
            dist_mean_temp.append(getMean(dist))
            # get difference magnitude in distance
            dist_diff_temp.append(getDiff(dist))
        dist_mean.append(dist_mean_temp)
        dist_diff.append(dist_diff_temp)

    utils.plot(os.path.join(out_dir, "p01_a-diff"), np.array(dist_diff), title="(max-min)/max", x_label="dimension",
               y_label="Euclidean Distance")
    utils.plot(os.path.join(out_dir, "p01_a-mean_distance"), np.array(dist_mean), title="mean distance",
               x_label="dimension",
               y_label="Mean Euclidean Distance")


if __name__ == '__main__':
    main(out_dir="../output", data_path="../data/wine.txt", sample_number=5)
