from utils import load_dataset, get_feature_vector1, get_feature_vector2, \
    plotPerColumnDistribution, \
    plotCorrelationMatrix, kmeans, pca, plotScatterMatrix, plotEigen, radar

from sklearn.preprocessing import minmax_scale
import pandas as pd

output_dir = "../output/p02/"
input_dir = "../data/b/"


def p02a(data_path, output_dir):
    X_raw, label = load_dataset(data_path, id_row="STBID",
                                exclude_row=["ID", "CODE", "ITEMCODE", "NAME", "PORTAL_VER",
                                             "STATUS", "TIME", "TYPE", "URL", "MD5"],
                                sample_number=100000)
    features = get_feature_vector1(X_raw, id_row='STBID', type_row='SHOW_TYPE',
                                   save_path=f"{output_dir}/1_features_100000.csv")
    features = features.drop(['id'], axis=1)
    features = pd.DataFrame(minmax_scale(features))
    features.columns = ['show_type1', 'show_type2', 'show_type3', 'show_type4']

    # 分析原始数据
    features.dataframeName = 'Favorite show record'
    # plotCorrelationMatrix(features, 10)
    # plotScatterMatrix(features, 20, 10)

    # f_2, var_2, cov_2 = pca(features, 2)
    f_3, var_3, cov_3 = pca(features, 3)
    # 绘制主成分
    # plotEigen(save_path=f"{output_dir}/a/eign_value.png", com=var_3)
    radar(cov_3[:3, :], features.columns, labels=['pc1', 'pc2', 'pc3'],
          title="principal component explained covariance",
          save_path=f"{output_dir}/a/pca_radar.png")

    # score = kmeans(f_2, 2, 10, save_path=f"{output_dir}/a/kmeans/2", metric=True, plot=True)
    # score = kmeans(f_3, 2, 10, save_path=f"{output_dir}/a/kmeans/3", metric=True, plot=True)
    # score = kmeans(features, 2, 10, save_path=f"{output_dir}/a/kmeans/raw", metric=True, plot=False)


def p02b(data_path, output_dir):
    X_raw, label = load_dataset(data_path, id_row="STBID", exclude_row=['L_CHANNEL_NAME'], sample_number=None)
    features = get_feature_vector2(X_raw, 'STBID', 'SID', save_path=f"{output_dir}/2_fearure.csv")
    features = features.drop(['id'], axis=1)
    features = pd.DataFrame(minmax_scale(features))
    features.columns = ['sum', 'mean', 'num', 'std', 'mostN to 80%']

    # 分析原始数据
    features.dataframeName = 'TV show record'
    # plotCorrelationMatrix(features, 10)
    # plotScatterMatrix(features, 20, 10)

    # f_2, var_2, cov_2 = pca(features, 2)
    f_3, var_3, cov_3 = pca(features, 3)
    # 绘制主成分
    # plotEigen(save_path=f"{output_dir}/b/eign_value.png", com=var_3)
    radar(cov_3[:3, :], features.columns, labels=['pc1', 'pc2', 'pc3'], title="principal component explained covariance",
          save_path=f"{output_dir}/b/pca_radar.png")

    # score = kmeans(f_2, 2, 10, save_path=f"{output_dir}/b/kmeans/2", metric=True, plot=False)
    # score = kmeans(f_3, 2, 10, save_path=f"{output_dir}/b/kmeans/3", metric=True, plot=False)
    # score = kmeans(features, 2, 10, save_path=f"{output_dir}/b/kmeans/raw", metric=True, plot=False)


if __name__ == '__main__':
    p02a(data_path=f"{input_dir}/1.csv", output_dir=output_dir)
    p02b(data_path=f"{input_dir}/2.csv", output_dir=output_dir)
