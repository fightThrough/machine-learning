import numpy as np
from MY_ALGORITHM_STUDY.utils.features.prepare_for_training import prepare_for_training

# arrayy = np.empty((5,0))
# print(arrayy)
# arrayy = np.arange(0,39)
# print(np.array_split(arrayy,2))
# raise ValueError('Can not generate polynomials for two sets with different number of rows')
# data1 = np.random.randint(0,5,(10,21),dtype="int")
# print(data1.shape)
# data2 = np.array_split(data1,2,axis=1)
# print(data2[0].shape)
class LinearRegression:
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data = True):
        (data_processed,features_mean,features_deviation) = prepare_for_training(data,labels,polynomial_degree,sinusoid_degree,normalize_data)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        self.num_features = self.data.shape[1]
        self.num_examples = self.data.shape[0]
        self.theta = np.zeros((self.num_features,1))

    def train(self,alpha=0.1,num_iterations=500):
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history
    def gradient_descent(self,alpha,num_iterations):#会迭代num_iterations次
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history
    def gradient_step(self,alpha):#实际执行模块，每一次更新参数
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha*(1/self.num_examples)*(np.dot(delta.T,self.data)).T
        self.theta = theta
    @staticmethod
    def hypothesis(data,theta):
        predictions = np.dot(data,theta)
        return predictions
    def cost_function(self,data,labels):
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)/self.num_examples
        return cost[0][0]
    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,labels,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        return self.cost_function(data_processed,labels)
    def predict(self,data):
        data_processed = prepare_for_training(data,self.labels,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        return predictions

