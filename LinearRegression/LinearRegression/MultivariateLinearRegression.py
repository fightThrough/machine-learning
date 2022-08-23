import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

# plotly.offline.init_notebook_mode()
from LinearRegression import LinearRegression
data = pd.read_csv("../data/world-happiness-report-2017.csv")

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
input_param_name_1 = "Economy..GDP.per.Capita."
input_param_name_2 = "Freedom"
output_param_name = "Happiness.Score"

x_train = train_data[[input_param_name_1,input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1,input_param_name_2]].values
y_test = test_data[[output_param_name]].values

plot_training_trace = go.Scatter3d(x=x_train[:,0].flatten(),y=x_train[:,1].flatten(),z=y_train.flatten(),name="training set",mode="markers",marker={"size":10,"opacity":1,"line":{"color":"rgb(255,255,255)","width":1}})
#markers代表点图
plot_test_trace = go.Scatter3d(x=x_test[:,0].flatten(),y=x_test[:,1].flatten(),z=y_test.flatten(),name="test set",mode="markers",marker={"size":10,"opacity":1,"line":{"color":"rgb(255,255,255)","width":1}})
plot_layout = go.Layout(title="Data set",scene={"xaxis":{"title":input_param_name_1},"yaxis":{"title":input_param_name_2},"zaxis":{"title":output_param_name}},margin={"l":0,"r":0,"b":0,"t":0})
plot_data = [plot_training_trace,plot_test_trace]
plot_figure = go.Figure(data=plot_data,layout=plot_layout)
plotly.offline.plot(plot_figure)

num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train,y_train,polynomial_degree=polynomial_degree,sinusoid_degree=sinusoid_degree)
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predict_num = 100
# x_prediction = np.linspace(x_train.min(),x_train.max(),predict_num).reshape(predict_num,1)
# y_prediction = linear_regression.predict(x_prediction)
x_min = x_train[:,0].min()
x_max = x_train[:,0].max()
y_min = x_train[:,1].min()
y_max = x_train[:,1].max()
x_axis = np.linspace(x_min,x_max,predict_num)
y_axis = np.linspace(y_min,y_max,predict_num)
x_prediction = np.zeros((predict_num*predict_num,1))
y_prediction = np.zeros((predict_num*predict_num,1))
x_y_index = 0
for x_index,x_value in enumerate(x_axis):
    for y_index,y_value in enumerate(y_axis):
        x_prediction[x_y_index] = x_value
        y_prediction[x_y_index] = y_value
        x_y_index += 1
x_y_prediction = np.hstack((x_prediction,y_prediction))
z_prediction = linear_regression.predict(x_y_prediction)

plot_predict_trace = go.Scatter3d(x=x_prediction.flatten(),y=y_prediction.flatten(),z=z_prediction.flatten(),name="prediction plane",mode="markers",marker={"size":1,"opacity":1,})
plot_layout = go.Layout(title="Data set",scene={"xaxis":{"title":input_param_name_1},"yaxis":{"title":input_param_name_2},"zaxis":{"title":output_param_name}},margin={"l":0,"r":0,"b":0,"t":0})
plot_data = [plot_training_trace,plot_test_trace,plot_predict_trace]
plot_figure = go.Figure(data=plot_data,layout=plot_layout)
plotly.offline.plot(plot_figure)

