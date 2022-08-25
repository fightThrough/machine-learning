from LinearRegression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/non-linear-regression-x-y.csv")
input_param_name = "y"
output_param_name = "x"

x_train = data[[input_param_name]].values
y_train = data[[output_param_name]].values
plt.scatter(x_train,y_train,label="train data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("scatter plot")
plt.show()

alpha = 0.1
num_iterations = 500
linearregression = LinearRegression(x_train,y_train,polynomial_degree=2,sinusoid_degree=2)
(theta,cost_functions) = linearregression.train(alpha,num_iterations)

predict_num = 100
input_param_min = x_train.min()
input_param_max = x_train.max()
input_predict = np.linspace(input_param_min,input_param_max,predict_num,dtype=float).reshape((predict_num,1))#shape 为（100，） 和（100，1）是不一样的，前者是一维数组
output_predict = linearregression.predict(input_predict)

plt.scatter(x_train,y_train,label="train data")
plt.scatter(input_predict,output_predict,label="predict line")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("scatter plot")
plt.show()