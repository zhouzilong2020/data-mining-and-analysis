import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sklearn.cluster as clu
import sklearn.metrics as met
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM


def get_feature_vector1(df, id_row, type_row, save_path):
    if os.path.exists(save_path):
        features = pd.read_csv(save_path)
        return features
    else:
        # 得到特征
        ids = set(df[id_row])
        types = list(set(df[type_row]))
        types.sort()
        type_dict = dict(zip(types, [i for i in range(len(types))]))
        features = np.zeros((len(ids), len(types)))
        per = 0
        period = int(len(ids) / 100)
        for i, id in enumerate(ids):
            if i % period == 0:
                per += 1
                print(f"{per - 1}-----100")
            selecteds = df[df[id_row] == id][type_row]
            for selected in selecteds:
                features[i][type_dict[selected]] += 1
        df = pd.DataFrame(features)
        df.to_csv(save_path)


def get_feature_vector2(df, id_name, channel_name, save_path, verbose=False):
    if os.path.exists(save_path):
        fd = open(save_path, encoding='gbk')
        data = pd.read_csv(fd)
        return data

    # 将cvs数据按照用户进行分类，转化为一个 （ 频道数，区域）特征向量
    ids = set(df[f'{id_name}'])

    features = []

    total_cnt = len(ids)
    print("extracting feature")

    for i, _id in enumerate(ids):
        selected = df.loc[(df[f'{id_name}'] == f'{_id}')]
        if len(selected) != 0:
            cnt = np.sum(selected.CNT)  # 总共收看次数
            mean = np.mean(selected.CNT)  # 平均每个频道收看次数
            num = np.count_nonzero(selected.CNT)  # 收看的频道数目
            var = np.sqrt(np.var(selected.CNT))  # 收看次数的方差

            sorted_CNT = np.sort(selected.CNT)
            tem_sum = 0
            mostN = 0  # 达到80%观看量的频道数目
            for j in range(len(sorted_CNT) - 1, -1, -1):
                tem_sum += sorted_CNT[j]
                mostN += 1
                if tem_sum / cnt > 0.8:
                    break

            # feature 包含一些频道的统计特性
            _feature = np.array([cnt, mean, num, var, mostN])
            features.append(_feature)
    features = np.array(features)
    # 保存
    df = pd.DataFrame(features)
    df.to_csv(save_path)

    return df


def load_dataset(data_path, id_row, exclude_row=[], sample_number=None, verbose=True):
    if verbose:
        print("loading dataset")

    # fd = open(data_path, encoding='gbk')
    if sample_number is not None:
        data = pd.read_csv(data_path, encoding='gbk', nrows=sample_number, error_bad_lines=False,
                           quoting=3)
    else:
        data = pd.read_csv(data_path, encoding='gbk', error_bad_lines=False, quoting=3)

    # 列索引问题
    data.rename(lambda o: str(o).strip('''"'''), axis=1, inplace=True)
    # # 解决引号问题
    # data.applymap(lambda o: o.replace('"', ''))

    # 删去主键为空的行
    data.dropna(axis=0, subset=[f"{id_row}"], inplace=True)
    data[f'{id_row}'] = data[f'{id_row}'].astype(str)
    # 按照主键排序
    data.sort_values(axis=0, ascending=True, by=id_row)
    # 删除不需要的行
    data.drop(exclude_row, axis=1, inplace=True)
    # 获取标签

    _label = data.keys()

    return data, _label


def radar(values, feature, labels, title, save_path):
    # 用于正常显示中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 用于正常显示符号
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    # 处理
    angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)[:]
    angles = np.concatenate((angles, [angles[0]]))
    # 绘图
    fig = plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)
    for value, label in zip(values, labels):
        value = np.concatenate((value, [value[0]]))
        # 绘制折线图
        ax.plot(angles, value, 'o-', linewidth=2, label=label)
        # 填充颜色
        ax.fill(angles, value, alpha=0.25)

    # 设置图标上的角度划分刻度，为每个数据点处添加标签
    ax.set_thetagrids(angles * 180 / np.pi, feature)
    # 设置雷达图的范围
    # min = np.min(values)
    # max = np.max(values)
    # scale = int(max - min / 10)
    # ax.set_ylim(min - scale, max + scale)

    plt.title(title)
    # 添加网格线
    ax.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def scatter2D(x, y, save_path, labels=None, title="Scatter Graph", x_label='PC_1', y_label='PC_2'):
    fig = plt.figure()
    if labels is not None:
        plt.scatter(x, y, c=labels, cmap="Set1")
    else:
        plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def scatter3D(x, y, z, title, save_path, labels=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title(title)
    if labels is not None:
        ax.scatter(x, y, z, c=labels, cmap="Set1")
    else:
        ax.scatter(x, y, z)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plotEigen(save_path, com, cum=None, title="PCA Decomposition", x_label='Proportion', y_label='i_th Eigen Value'):
    fig = plt.figure()
    plt.bar(range(com.shape[0]), com, alpha=0.5, align='center',
            label='Eigen Value')
    if cum is None:
        cum = []
        cum.append(com[0])
        for i in range(1, com.shape[0]):
            cum.append(cum[-1] + com[i])
        cum = np.array(cum)

    plt.step(range(cum.shape[0]), cum, where='mid',
             label='Cumulative Eigen Value')
    plt.ylabel(ylabel=y_label)
    plt.xlabel(xlabel=x_label)
    plt.title(label=title)
    plt.legend(loc='best')
    plt.savefig(save_path)
    plt.close(fig)


def scoreLine(score_type, scores, cluster_num, title, save_path):
    '''
    绘制聚类评价指标图
    :param score_type:评价指标的名称
    :param scores:一个n*line_num*score_num的矩阵
    :param cluster_num:聚类数目，与每一个得分相对应
    :return:
    '''
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig = plt.figure()
    colors = ["deeppink", "darkblue", "goldenrod"]
    markers = ['o', '+', '*']
    line_styles = [':', '--', '-.']

    scores = minmax_scale(scores)

    plt.title(title)
    plt.xlabel(u'cluster_num', fontsize=14)
    plt.ylabel(u'index', fontsize=14)

    for i, _type in enumerate(score_type):
        plt.plot(cluster_num,
                 scores[:, i],
                 label=_type,
                 marker=markers[i],
                 linestyle=line_styles[i],
                 color=colors[i])
    plt.xticks(cluster_num)
    plt.legend()
    plt.savefig(f"{save_path}/score.png", bbox_inches='tight')
    plt.close(fig)


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[
        col] < 50]]  # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show(bbox_inches='tight')


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns')  # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show(bbox_inches='tight')


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                          va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


def kmeans(X, bg, ed, save_path, metric=False, plot=False, verbose=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    scores = []
    if verbose:
        print("strat kmeans")
    for i in range(bg, ed):
        model = clu.KMeans(n_clusters=i).fit(X)
        labels = model.labels_
        if plot:
            if X.shape[1] == 2:
                scatter2D(X[:, 0], X[:, 1], save_path=f"{save_path}/cluser_{i}.png", labels=labels,
                          title=f"k_means_{i}")
            if X.shape[1] == 3:
                scatter3D(X[:, 0], X[:, 1], X[:, 2], save_path=f"{save_path}/cluser_{i}.png", labels=labels,
                          title=f"k_means_{i}")
        if metric:
            score = get_metric(X, labels)
            scores.append(score)

        center = model.cluster_centers_
        pd.DataFrame(center).to_csv(f"{save_path}/cluster_center_{i}.csv")
        if verbose and metric:
            print(f"n_clusters = {i}, score:{score}")

    if metric:
        scoreLine(['silhouette', 'calinski_harabasz'], scores, np.array([i for i in range(bg, ed)]),
                  title="cluster index",
                  save_path=save_path)
    return scores


def dbscan(X, save_path, eps=0.3, ms=10, metric=False, plot=False, verbose=True):
    scores = []
    if verbose:
        print("strat dbscan")

    model = clu.DBSCAN(eps=eps, min_samples=ms).fit(X)
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    labels = model.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if verbose:
        print(f'Estimated number of clusters: {n_clusters}')
        print(f'Estimated number of noise points:{n_noise}')

    if plot:
        scatter3D(X[:, 0], X[:, 1], X[:, 2], title=f"dbscan_eps={eps}_ms={ms}",
                  save_path=f"{save_path}/dbscan_{n_clusters}",
                  labels=labels)
    if metric:
        score = get_metric(X, labels)
        scores.append(score)

    return scores


def pca(X, n, verbose=True):
    p = PCA(n_components=n)
    X_pca = p.fit_transform(X)

    if verbose:
        print(p.explained_variance_ratio_)
        print(np.sum(p.explained_variance_ratio_))
    return X_pca, p.explained_variance_ratio_, p.get_covariance()


def gmm(X, bg, ed, save_path, metric=False, plot=False, verbose=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    scores = []
    if verbose:
        print("strat gmm")
    for i in range(bg, ed):
        labels = GMM(n_components=i).fit(X).predict(X)
        if plot:
            if X.shape[1] == 2:
                scatter2D(X[:, 0], X[:, 1], save_path=f"{save_path}/cluser_{i}.png", labels=labels,
                          title=f"k_means_{i}")
            if X.shape[1] == 3:
                scatter3D(X[:, 0], X[:, 1], X[:, 2], save_path=f"{save_path}/cluser_{i}.png", labels=labels,
                          title=f"k_means_{i}")
        if metric:
            score = get_metric(X, labels)
            scores.append(score)

        if verbose and metric:
            print(f"n_clusters = {i}, score:{score}")

    if metric:
        scoreLine(['silhouette', 'calinski_harabasz'], scores, np.array([i for i in range(bg, ed)]),
                  title="cluster index",
                  save_path=save_path)
    return scores


def get_metric(X, labels):
    si = met.silhouette_score(X, labels)
    ca = met.calinski_harabasz_score(X, labels)
    return [si, ca]
