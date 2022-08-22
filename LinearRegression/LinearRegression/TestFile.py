from LinearRegression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/world-happiness-report-2017.csv")
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
input_param_name = "Economy..GDP.per.Capita."
output_param_name = "Happiness.Score"
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

plt.scatter(x_train,y_train,label="train data")
plt.scatter(x_test,y_test,label="test data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("Happy")
plt.legend()
plt.show()

num_iterations = 500
learning_rate = 0.01
linear_regression = LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(alpha=learning_rate,num_iterations=num_iterations)
plt.plot(range(num_iterations),cost_history)
plt.xlabel("Iter")
plt.ylabel("cost")
plt.title("GD")
plt.show()

predictions_num = 100
x_prediction = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_prediction = linear_regression.predict(x_prediction)

plt.scatter(x_train,y_train,label="train data")
plt.scatter(x_test,y_test,label="test data")
plt.scatter(x_prediction,y_prediction,label="predict data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()


