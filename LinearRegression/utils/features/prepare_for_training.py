import numpy as np
from .normalize import normalize
from .sinusoid import sinusoid_f
from .polynomial import polynomial
def prepare_for_training(data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data = True):
    num_examples = data.shape[0]#矩阵行数，即为数据组数,多少个样本
    data_processed = np.copy(data)#copy一份数据，不能修改原数据
    #预处理
    #ctrl+b 可以跳转到函数具体用法与实例
    data_normalize = data_processed
    if normalize_data:#需要标准化
        (data_normalize,data_mean,data_deviation)= normalize(data_processed)
        data_processed = data_normalize#标准化
    if sinusoid_degree > 0:#可以sin特征变换的情况，变换方式 sin（x）
        data_sinusoid = sinusoid_f(data_normalize,sinusoid_degree)
        data_processed = np.concatenate((data_processed,data_sinusoid),axis=1)
    if polynomial_degree > 0:#可以pol特征变换的情况，变换方式x1,x2,x1*x2,x1^2,x2^2..
        data_polynomial = polynomial(data_normalize,polynomial_degree,normalize_data)
        data_processed = np.concatenate((data_processed,data_polynomial),axis=1)
    data_processed = np.hstack((np.ones((num_examples,1)),data_processed))#加一列1,多个参数传参要加括号
    return data_processed,data_mean,data_deviation


