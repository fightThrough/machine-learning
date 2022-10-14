import numpy as np
from sklearn.linear_model import LogisticRegression #参数C表示正则化惩罚力度
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris
iris = load_iris()
# # print(iris.DESCR)#数据集的描述
# # print(iris.keys())
# # X = iris['data'][:,3:]#二维数组
# # print(X)
# # Y = (iris['target'] == 2).astype(int)#变化标签
# # print(Y)
# # log_reg = LogisticRegression()
# # log_reg.fit(X,Y)
# # x_new = np.linspace(0,3,1000).reshape(-1,1)#-1表示自动计算行数
# # y_proba = log_reg.predict_proba(x_new)#打印概率
# # print(y_proba)
# # decision_boundary = x_new[y_proba[:,0]>=0.5][-1]
# # plt.plot([decision_boundary,decision_boundary],[-2,2],'k--')
# # plt.plot(x_new,y_proba[:,0],'g--',label='Virginica')
# # plt.plot(x_new,y_proba[:,1],'b--',label='Not Virginica')
# # plt.arrow(decision_boundary,-1.5,-0.5,0,head_width=0.05,head_length=0.1)
# # plt.arrow(decision_boundary,1.5,0.5,0,head_width=0.05,head_length=0.1)
# # plt.legend()
# # plt.show()
# # print(help(plt.text))#添加文本
# X = iris['data'][:,(2,3)]
# Y = (iris['target']==2).astype(int)
# # log_reg.fit(X,Y)
# x0,x1 = np.meshgrid(np.linspace(2.9,7,500).reshape(-1,1),np.linspace(0.8,2.7,200).reshape(-1,1))
# # print(x0)
# # print(x1)
# x2 = np.c_[x0.ravel(),x1.ravel()]
# # print(x2)#笛卡尔乘积，坐标棋盘
# print(x2.shape)
# y_probe = log_reg.predict(x2)
# plt.figure(figsize=(10,4))
# plt.plot(X[Y==0,0],X[Y==0,1],'bs')
# plt.plot(X[Y==1,0],X[Y==1,1],'g^')
# zz = y_probe[:,0].reshape(x0.shape)
# contour = plt.contour(x0,x1,zz,cmap=plt.cm.brg)
# plt.clabel(contour,inline = 1)

X = iris['data'][:,(2,3)]
Y = iris['target']
softmax_reg = LogisticRegression(multi_class = 'multinomial',solver='newton-cg')
softmax_reg.fit(X,Y)
print(softmax_reg.predict_proba([[5,2]]))#得到三个位置概率值
y_proba = softmax_reg.predict_proba(X)
y_predict = softmax_reg.predict(X)
plt.axis([0,7,0,3.5])
plt.plot(X[Y==0,0],X[Y==0,1],'g.',label='L1')
plt.plot(X[Y==1,0],X[Y==1,1],'b*',label='L2')
plt.plot(X[Y==2,0],X[Y==2,1],'k^',label='L3')
x0,x1 = np.meshgrid(np.linspace(0,7,200).reshape(-1,1),np.linspace(0,3.5,100).reshape(-1,1))
x2 = np.c_[x0.ravel(),x1.ravel()]
plt.contour(x0,x1,softmax_reg.predict_proba(x2)[:,1].reshape(x0.shape))
plt.show()
