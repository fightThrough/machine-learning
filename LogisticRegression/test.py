from LogisticRegression import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

load_iris = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']
print(load_iris.shape)

x_axis = 'petal_length'
y_axis = 'petal_width'
for iris_type in iris_types:
    plt.scatter(load_iris[x_axis][load_iris['class'] == iris_type],#一堆横坐标
                load_iris[y_axis][load_iris['class'] == iris_type],
                label = iris_type)
plt.legend()

num_examples = load_iris.shape[0]
x_train = load_iris[[x_axis,y_axis]].values.reshape((num_examples,2))
y_train = load_iris['class'].values.reshape((num_examples,1))

max_iterations = 10000
polynomial_degree=0
sinusoid_degree=0
log_reg = LogisticRegression(x_train,y_train,polynomial_degree, sinusoid_degree, normalize_data=False)
(thetas,cost_histories) = log_reg.train(max_iterations)
labels = np.unique(y_train)
plt.plot(range(len(cost_histories[0])),cost_histories[0],'r--',label=labels[0])
plt.plot(range(len(cost_histories[1])),cost_histories[1],'b--',label=labels[1])
plt.plot(range(len(cost_histories[2])),cost_histories[2],'g--',label=labels[2])
plt.legend()
plt.show()

x_min = np.min(x_train[:,0])
x_max = np.max(x_train[:,0])
y_min = np.min(x_train[:,1])
y_max = np.max(x_train[:,1])
samples = 150
X = np.linspace(x_min,x_max,samples)
Y = np.linspace(y_min,y_max,samples)
Z_1 = np.zeros((samples,samples))
Z_2 = np.zeros((samples,samples))
Z_3 = np.zeros((samples,samples))

for x_index,x in enumerate(X):
    for y_index,y in enumerate(Y):
        data = np.array([[x,y]])
        prediction = log_reg.predict(data)[0][0]
        if(prediction == 'SETOSA'):
            Z_1[x_index][y_index] = 1
        elif(prediction == 'VERSICOLOR'):
            Z_2[x_index][y_index] = 1
        else:
            Z_3[x_index][y_index] = 1
for iris_type in iris_types:
    plt.scatter(x_train[(y_train == iris_type).flatten(),0], # 一堆横坐标
                x_train[(y_train == iris_type).flatten(),1],
                label=iris_type)
plt.contour(X,Y,Z_1)#等高线画法
plt.contour(X,Y,Z_2)
plt.contour(X,Y,Z_3)
plt.show()




