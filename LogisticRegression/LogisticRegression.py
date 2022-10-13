#逻辑回归是分类算法，解决经典二分类
#先用简单的模型再用复杂的
#非线性决策边界
#sigmoid：g(z)=1/(1+e**(-z)) 任意的输入映射到[0,1]中

import numpy as np
from scipy.optimize import minimize
from MY_ALGORITHM_STUDY_LINEAR_REGRESSION.utils.features.prepare_for_training import prepare_for_training
from MY_ALGORITHM_STUDY_LINEAR_REGRESSION.utils.hypothesis.sigmoid import sigmoid

class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, labels, polynomial_degree,
                                                                                   sinusoid_degree, normalize_data=True)
        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        self.num_features = self.data.shape[1]
        self.theta = np.zeros((self.num_features, self.unique_labels.shape[0]))#新的theta

    def train(self, max_iterations=1000):
        cost_history = []
        cost_histories = []
        num_features = self.num_features
        for label_index,unique_label in enumerate(self.unique_labels):#对于每一类不同来处理
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features,1))
            current_labels = (self.labels == unique_label).astype(float)#更改标签
            (current_theta,cost_history) = LogisticRegression.gradient_descent(self.data,current_labels,current_initial_theta,max_iterations)
            cost_histories.append(cost_history)
            self.theta[label_index] = current_theta.T
        return self.theta,cost_history
    @staticmethod
    def gradient_descent(data,labels,current_initial_theta,max_iterations):  # 会迭代num_iterations次
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
            #要优化的目标
            fun=lambda current_theta:LogisticRegression.cost_function(data,labels,current_theta.reshape(num_features,1)),
            #初始化的权重参数
            x0=current_initial_theta.reshape(num_features,1),
            #优化策略
            method='CG',
            #梯度下降迭代计算公式
            jac=lambda current_theta:LogisticRegression.gradient_step(data,labels,current_theta.reshape(num_features,1)),
            #记录结果
            callback = lambda current_theta:cost_history.append(LogisticRegression.cost_function(data,labels,current_theta.reshape(num_features,1))),
            #迭代次数
            options={'maxiter':max_iterations}
        )
        print(result)
        if not result.success:
            raise ArithmeticError('can not minimize cost function '+result.message)
        theta = result.x.reshape(num_features,1)
        return theta,cost_history
    @staticmethod
    def cost_function(data,labels,theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data,theta)
        y_is_set_cost = np.dot(labels[labels == 1].T,np.log(predictions[labels == 1]))
        y_is_not_cost = np.dot(1 - labels[labels == 0].T,np.log(1 - predictions[labels == 0]))
        cost = (-1/num_examples)*(y_is_not_cost + y_is_not_cost)
        return cost
    @staticmethod
    def hypothesis(data,theta):
        return sigmoid(np.dot(data,theta))

    @staticmethod
    def gradient_step(data,labels,theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1/num_examples)*np.dot(data.T,label_diff)
        return gradients.T.flatten()

    def predict(self,data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        prob = LogisticRegression.hypothesis(data_processed,theta=self.theta)
        max_prob_index = np.argmax(prob,axis=1)#每一行最大的
        class_prediction = np.empty(max_prob_index.shape,dtype=object)
        for index,label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction
