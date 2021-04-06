import numpy as np
import utils
import os


class PCA:
    def __init__(self, X):
        self.X = X
        self.lamda = None
        self.P = None
        self.com = None
        self.cum = None
        self.T = None

    def SVDDecompose(self):
        B = np.linalg.svd(self.X, full_matrices=False)
        self.lamda = B[1]
        self.P = B[2].T
        self.T = B[0] * B[1]
        # cumulative proportion
        cum = self.lamda.cumsum() / self.lamda.sum() * 100
        self.com = np.array(self.lamda)
        self.cum = cum

    def PCAdecompose(self, k):
        T = self.T[:, :k]
        P = self.P[:, :k]
        return T, P

    def plot_eigen_value(self, file_name):
        utils.plot_eigen(file_name + 'Eigen_Value', self.com, self.cum)

    def plot_scatter(self, file_name, X, y):
        utils.plot_scatter(file_name + 'Result', X, y)


class LDA:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.ndim = X.shape[1]
        self.mean_vecs = None
        self.S_w = None
        self.com = None
        self.cum = None
        self.S_b = None
        self.eigen_pairs = None

    def mean(self):
        label_cnt = np.unique(self.y).shape[0]
        mean_temp = []
        for label in range(1, 1 + label_cnt):
            mean_temp.append(np.mean(self.X[self.y == label], axis=0))
        self.mean_vecs = np.array(mean_temp)

    def Sw(self):
        label_cnt = np.unique(self.y).shape[0]
        S_w = np.zeros((self.ndim, self.ndim))
        # mean vector
        for label, mv in zip(range(1, 1 + label_cnt), self.mean_vecs):
            # calculate divergence inside each class
            class_scatter = np.zeros((self.ndim, self.ndim))
            for row in self.X[self.y == label]:
                # row vector
                row, mv = row.reshape(self.ndim, 1), mv.reshape(self.ndim, 1)
                class_scatter += (row - mv).dot((row - mv).T)

            S_w += class_scatter
        self.S_w = S_w

    def Sb(self):
        # overall mean value
        mean_overall = np.mean(self.X, axis=0)
        S_b = np.zeros((self.ndim, self.ndim))
        # calculate mean value for each group
        for i, mean_vec in enumerate(self.mean_vecs):
            n = self.X[self.y == i + 1, :].shape[0]
            # column vector
            mean_vec = mean_vec.reshape(self.ndim, 1)
            mean_overall = mean_overall.reshape(self.ndim, 1)
            # divergence between class
            S_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
        self.S_b = S_b

    def eigen_value(self):
        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(self.S_w).dot(self.S_b))
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                       for i in range(len(eigen_vals))]

        self.eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        tot = sum(eigen_vals.real)
        # calculate proportion
        self.com = np.array([(i / tot) for i in sorted(eigen_vals.real, reverse=True)])
        # calculate accumulation
        self.cum = np.array(np.cumsum(self.com))

    def plot_eigen_value(self, file_name):
        utils.plot_eigen(file_name + 'Eigen_Value', self.com, self.cum)

    def plot_scatter(self, file_name, X, y):
        utils.plot_scatter(file_name + 'Result', X, y)

    def LDADecompose(self, X):
        w = np.hstack((self.eigen_pairs[0][1][:, np.newaxis].real,
                       self.eigen_pairs[1][1][:, np.newaxis].real))
        return X.dot(w)


def main(out_dir, data_path, exclude_dim=[]):
    # prepare data
    data = utils.load_dataset(data_path=data_path, exclude_dim=exclude_dim, sample_number=np.inf)
    label = data[:, 0]
    data = data[:, 1:]
    data = utils.normalize(data)
    # PCA Decomposition
    pca = PCA(data)
    pca.SVDDecompose()
    # get eigen value
    pca.plot_eigen_value(os.path.join(out_dir, "p01_b_PCA_Decomposition_"))
    # get result value
    X = pca.PCAdecompose(2)[0]
    pca.plot_scatter(os.path.join(out_dir, "p01_b_PCA_Decomposition_"), X, label)

    # LDA Decomposition
    lda = LDA(data, label)
    lda.mean()
    lda.Sb()
    lda.Sw()
    lda.eigen_value()
    lda.plot_eigen_value(os.path.join(out_dir, "p01_b_LDA_Decomposition_"))
    X = lda.LDADecompose(data)
    lda.plot_scatter(os.path.join(out_dir, "p01_b_LDA_Decomposition_"), X, label)


if __name__ == '__main__':
    main(out_dir="../output", data_path="../data/wine.txt")
