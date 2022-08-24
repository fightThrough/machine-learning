import numpy as np
from .normalize import normalize
def polynomial(dataset,polynomial_degree,normalize_data = False):
    '''
    data1 = np.random.randint(0,5,(10,20),dtype="int")
    print(data1.shape)
    data2 = np.array_split(data1,2,axis=1)
    print(data2[0].shape)
    output:
    (10, 20)
    (10, 10)
    竖向切割
    :param dataset:
    :param polynomial_degree:
    :return:
    '''
    features_split = np.array_split(dataset,2,axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]
    (data1_shapex,data1_shapey) = dataset_1.shape
    (data2_shapex,data2_shapey) = dataset_2.shape
    if(data1_shapex != data2_shapex):
        raise ValueError('Can not generate polynomials for two sets with different number of rows')
    if(data1_shapey == 0 and data2_shapey == 0):
        raise ValueError('Can not generate polynomials for two sets with no columns')
    if data1_shapey == 0:
        dataset_1 = dataset_2
    if data2_shapey == 0:
        dataset_2 = dataset_1
    num_features = data1_shapey if data1_shapey < data2_shapey else data2_shapey
    dataset_1 = dataset_1[:,:num_features]
    dataset_2 = dataset_2[:,:num_features]
    polynomials = np.empty((data1_shapex,0))
    for i in range(1,polynomial_degree+1):
        for j in range(i+1):
            polynomials_features = (dataset_1**(i-j))*(dataset_2**(j))
            polynomials = np.concatenate((polynomials,polynomials_features),axis=1)
    if normalize_data:
        polynomials = normalize(polynomials)[0]
    return polynomials
