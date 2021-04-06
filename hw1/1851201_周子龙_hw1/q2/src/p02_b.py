import numpy as np
import utils
import numpy as np
import scipy.stats


def cos_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * (np.linalg.norm(x2)))


def KL_divergence(p, q):
    return scipy.stats.entropy(p, q)


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


methods = [cos_similarity, KL_divergence, JS_divergence]


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
    methods_name = ['cos_similarity', 'KL_divergence', 'JS_divergence']
    utils.plot(out_dir + 'similarity', dis, methods_name, 'distance comparison', 'method', 'distance')
    pass


if __name__ == '__main__':
    main(out_dir="../output/", data_path="../data/wine.txt")

举例来说，100个平均分布的点能把一个单位区间以每个点距离不超过0.01采样；而当维度增加到10后，如果以相邻点距离不超过0.01小方格采样一单位超正方体，则需要1020 个采样点:所以，这个10维的超正方体也可以说是比单位区间大1018倍。