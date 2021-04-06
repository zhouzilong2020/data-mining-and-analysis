import numpy as np
import utils


def e_distance(x1, x2):
    return np.sqrt(np.sum(np.power((x1 - x2), 2)))


def mini_distance(x1, x2):
    dim = x1.shape[0]
    return np.power(np.sum(np.power(np.abs(x1 - x2), dim)), 1 / dim)


def chebyshev_distance(x1, x2):
    return np.abs(x1 - x2).max()


methods = [e_distance, mini_distance, chebyshev_distance]


def getDistance(data):
    result = []
    for method in methods:
        dis = []
        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    dis.append(method(data[i], data[j]))
        result.append(dis)
    return np.array(result)


def main(out_dir, data_path):
    data = utils.load_dataset(data_path)
    dis = getDistance(data)
    methods_name = ['e_distance', 'mini_distance', 'chebyshev_distance']
    utils.plot(out_dir + 'distance', dis, methods_name, 'distance comparison', 'method', 'distance')
    pass


if __name__ == '__main__':
    main(out_dir="../output/", data_path="../data/wine.txt")
