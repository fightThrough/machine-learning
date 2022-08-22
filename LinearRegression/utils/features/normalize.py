import numpy as np
def normalize(features):
    features_normalized = np.copy(features).astype(float)
    features_mean = features_normalized.mean(0)#求平均值，0代表对每列的这个特征求平均值，返回一个一维数组，原矩阵多少列（特征）返回数组多少元素
    # features_mean = np.mean(features,0)
    features_deviation = np.std(features,0)#求标准差
    if features.shape[0]>1:
        features_normalized -= features_mean#数据组数只有一组的没有意义，0组的前面已经排除
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation
    return features_normalized,features_mean,features_deviation