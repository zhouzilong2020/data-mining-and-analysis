import pandas as pd
import os
from utils import load_dataset, scatter2D, plotPerColumnDistribution, plotCorrelationMatrix, plotScatterMatrix, \
    kmeans, pca, gmm, scatter3D, dbscan, scoreLine, plotEigen, radar
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.preprocessing import minmax_scale

data_path = "../data/a/diabetic_data.csv"
output_path = "../output/p01/"

race_dict = {'Caucasian': 0, 'AfricanAmerican': 1, 'Hispanic': 2, '?': 3, 'other': 3, 'default': 3}
gender_dict = {'Female': 0, 'Male': 1, 'default': 2}
age_dict = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6,
            '[70-80)': 7, '[80-90)': 8, '[90-100)': 9, 'default': 10}
metformin_dict = {'Steady': 0, 'No': 1, 'default': 2}
insulin_dict = {'Steady': 0, 'No': 1, 'Down': 2, 'Up': 3, 'default': 4}
change_dict = {'Ch': 0, 'No': 1, 'default': 2}
diabetesMed_dict = {'Yes': 0, 'No': 1}

mapping_dict = {'race': race_dict, 'gender': gender_dict, 'age': age_dict, \
                'metformin': metformin_dict, 'insulin': insulin_dict, \
                'change': change_dict, 'diabetesMed': diabetesMed_dict}


def is_numeric(row):
    return str(row['diag_1']).isnumeric() and str(row['diag_2']).isnumeric() and str(row['diag_3']).isnumeric()


def get_mapping_attribute(row_label, _dict):
    if _dict.__contains__(row_label):
        return _dict[row_label]
    else:
        return _dict['default']


def preprocess(df):
    for i in range(df.shape[0]):
        # 直接删去异常值
        if not is_numeric(df.loc[i, :]):
            df = df.drop([i])
            continue
        # 将原始数据进行转换
        for row_label, _dict in zip(mapping_dict.keys(), mapping_dict.values()):
            df.loc[i, row_label] = get_mapping_attribute(df.loc[i, row_label], _dict)
    # 将中间的数据转换为int
    df.loc[:, ['diag_1', 'diag_2', 'diag_3']].astype('int')
    return df


def delete_duplicate(df, id_row):
    df.drop_duplicates([f'{id_row}'])


def plotRadar():
    values = pd.read_csv("../output/p01/kmeans/3/cluster_center_3.csv").to_numpy()[:, 1:]
    featue_label = ["pc1", "pc2", "pc3"]
    labels = ["cls1", "cls2", "cls3"]

    radar(values=values, feature=featue_label, labels=labels,
          title="diabetes cluster radar graph",
          save_path=f"{output_path}/radar.png")


def getCov3():
    if not os.path.exists(path=f"{output_path}/processed.csv"):
        X_raw, label = load_dataset(data_path=data_path, id_row="patient_nbr",
                                    exclude_row=['encounter_id', 'weight', 'payer_code', 'medical_specialty',
                                                 'number_outpatient', 'number_emergency', 'max_glu_serum',
                                                 'A1Cresult', 'repaglinide', 'nateglinide', 'chlorpropamide',
                                                 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                                                 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                                                 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                                                 'glyburide-metformin', 'glipizide-metformin',
                                                 'glimepiride-pioglitazone',
                                                 'metformin-rosiglitazone', 'metformin-pioglitazone', 'readmitted'
                                                 ]
                                    )
        X = preprocess(X_raw)
    else:
        # 删去序号
        X, label = load_dataset(data_path=f"{output_path}/processed.csv", id_row="patient_nbr",
                                exclude_row=['id',
                                             'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                                             'diag_1', 'diag_3', 'diag_2',
                                             ])
        X.drop_duplicates(['patient_nbr'], inplace=True)
        X.drop(labels=['patient_nbr'], axis=1, inplace=True)
    label = X.keys()
    X_pca_3, p_3, cov_3 = pca(X, 3)
    return cov_3, label


def main():
    if not os.path.exists(path=f"{output_path}/processed.csv"):
        X_raw, label = load_dataset(data_path=data_path, id_row="patient_nbr",
                                    exclude_row=['encounter_id', 'weight', 'payer_code', 'medical_specialty',
                                                 'number_outpatient', 'number_emergency', 'max_glu_serum',
                                                 'A1Cresult', 'repaglinide', 'nateglinide', 'chlorpropamide',
                                                 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                                                 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                                                 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                                                 'glyburide-metformin', 'glipizide-metformin',
                                                 'glimepiride-pioglitazone',
                                                 'metformin-rosiglitazone', 'metformin-pioglitazone', 'readmitted'
                                                 ]
                                    )
        X = preprocess(X_raw)
    else:
        # 删去序号
        X, label = load_dataset(data_path=f"{output_path}/processed.csv", id_row="patient_nbr",
                                exclude_row=['id',
                                             'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                                             'diag_1', 'diag_3', 'diag_2',
                                             ])
        X.drop_duplicates(['patient_nbr'], inplace=True)
        X.drop(labels=['patient_nbr'], axis=1, inplace=True)

    X.dataframeName = 'diabetic_data.csv'
    # 分析原始数据
    plotCorrelationMatrix(X, 10)
    plotScatterMatrix(X, 20, 10)

    # 预处理
    X = minmax_scale(X)

    # 降维可视化
    X_pca_2, p_2, cov_2 = pca(X, 2)
    X_pca_3, p_3, cov_3 = pca(X, 3)
    scatter2D(X_pca_2[:, 0], X_pca_2[:, 1],
              save_path=f"{output_path}/(nal)pca_2.png", title="X_PCA_2_components")
    scatter3D(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2],
              save_path=f"{output_path}/(nal)pca_3.png", title="X_PCA_3_components", )

    # 绘制主成分分析碎石图
    a, b = pca(X, 10)
    plotEigen(save_path=f"{output_path}/eign_value.png", com=b)
    cluser_range = (2, 10)
    # 预测
    score_raw = kmeans(X, cluser_range[0], cluser_range[1], metric=True, plot=False,
                       save_path=f"{output_path}/kmeans/raw/")
    score_2 = kmeans(X_pca_2, cluser_range[0], cluser_range[1], metric=True, plot=False,
                     save_path=f"{output_path}/kmeans/2/")
    score_3 = kmeans(X_pca_3, cluser_range[0], cluser_range[1], metric=True, plot=False,
                     save_path=f"{output_path}/kmeans/3/")

    # score = dbscan(X_pca, eps=0.3, ms=10, metric=False, plot=True, save_path=f"{output_path}/dbscan")
    # gmm(X_pca_3, cluser_range[0], cluser_range[1], metric=True, plot=True,
    #     save_path=f"{output_path}/gmm/3/")


if __name__ == '__main__':
    cov, feature = getCov3()
    cov = cov[:3, :]
    radar(cov, feature, labels=['pc1', 'pc2', 'pc3'], title="principal component explained covariance",
          save_path=f"{output_path}/pca_radar.png")
# a
