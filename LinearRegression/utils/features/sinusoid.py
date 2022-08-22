import numpy as np
def sinusoid_f(datasets,sinusoids_degree):
    num_examples = datasets.shape[0]
    sinusoid_array = np.empty((num_examples,0))#n行0列的数据，可以变成新特征
    for degree in range(1,sinusoids_degree+1):#左开右闭
        sinusoid_features = np.sin(degree*datasets)
        sinusoid_array = np.concatenate((sinusoid_array,sinusoid_features),axis=1)#横着加
    return sinusoid_array