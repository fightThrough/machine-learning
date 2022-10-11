#关于梯度下降
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures#流水线车间
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error#均方误差
from sklearn.model_selection import train_test_split#训练集与测试集的分割
from sklearn.linear_model import Ridge#岭回归正则化

# X=np.random.rand(100,1)
# Y=3+4*X+np.random.rand(100,1)
# plt.plot(X,Y,"b.")
# plt.xlim([0,1])
# plt.ylim([2,8])
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
# X_b = np.c_[(np.ones((100,1)),X)]
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
# print(theta_best)
# X_new = np.array([[0],[1]])
# X_new_b = np.c_[(np.ones((2,1))),X_new]
# Y_pred = X_new_b.dot(theta_best)
# print(Y_pred)
# plt.plot(X_new,Y_pred,"r--")
# plt.plot(X,Y,"b.")
# plt.axis([0,1,2,8])
# plt.show()
#
# lin_reg = LinearRegression()
# lin_reg.fit(X,Y)
# print(lin_reg.coef_)#权重参数
# print(lin_reg.intercept_)#偏置参数 与上面theta求值相同
#
# eta = 0.1#学习率
# n_iterations = 1000#最大迭代次数
# m = 100 #100个样本
# theta = np.random.rand(2,1)
# for iteration in range(n_iterations):
#     gradients = 2/m* X_b.T.dot(X_b.dot(theta)-Y)#批量梯度下降
#     theta = theta-eta*gradients
# print(theta)

# #批量梯度下降
# theta_path_pgd=[]
# def plot_gradient_descent(theta,eta,theta_path = None):
#     m = len(X_b)
#     plt.plot(X,Y,"b.")
#     n_iterations = 1000
#     for iteration in range(n_iterations):
#         Y_p = X_new_b.dot(theta)
#         plt.plot(X_new,Y_p,'r-')
#         gradients = 2/m*X_b.T.dot(X_b.dot(theta)-Y)
#         theta = theta-eta*gradients
#         theta_path_pgd.append(theta)
#     print("批量梯度下降，学习率={},theta={}".format(eta,theta))
#     plt.xlabel("x")
#     plt.axis([0,1,2,8])
#     plt.title("eta={}".format(eta))
# theta = np.random.rand(2,1)
# plt.figure(figsize=(10,4))
# plt.subplot(131)
# plot_gradient_descent(theta,eta=0.02)
# plt.subplot(132)
# plot_gradient_descent(theta,eta=0.05)
# plt.subplot(133)
# plot_gradient_descent(theta,eta=0.1)
# plt.show()

# #随机梯度下降
# theta_path_sgd=[]
# m=len(X_b)
# n_epochs=50
# t0=5
# t1=50
# def learning_schedule(t):
#     return t0/(t1+t)
# for epoch in range(n_epochs):
#     for i in range(m):
#         if epoch < 10 and i < 10:
#             Y_p = X_new_b.dot(theta)
#             plt.plot(X_new,Y_p,'r-')
#         random_index = np.random.randint(m)
#         xi=X_b[random_index:random_index+1]
#         yi=Y[random_index:random_index+1]
#         gradients = 2*xi.T.dot(xi.dot(theta)-yi)
#         eta = learning_schedule(n_epochs*m+i)
#         theta = theta - eta*gradients
#         theta_path_sgd.append(theta)
# print("随机梯度下降，theta={}".format(theta))
# plt.plot(X,Y,'b.')
# plt.axis([0,1,2,8])
# plt.show()

# #小批量梯度下降 minibatch
# t=0
# theta_path_mgd=[]
# n_epochs=100
# minibatch=16
# eta=0.02
# theta = np.random.randn(2,1)
# np.random.seed(0)#设置随机种子，结果每次不变
# for epoch in range(n_epochs):#每一个epoch需要洗牌
#     shuffled_index = np.random.permutation(m)
#     X_b_shuffled = X_b[shuffled_index]
#     Y_shuffled = Y[shuffled_index]
#     for i in range(0,m,minibatch):
#         t+=1
#         xi = X_b_shuffled[i:i+minibatch]
#         yi = Y_shuffled[i:i+minibatch]
#         gradients = 2/minibatch*xi.T.dot(xi.dot(theta)-yi)
#         theta = theta-eta*gradients
#         theta_path_mgd.append(theta)
# print("minibatch梯度下降，theta={}".format(theta))

# # plt.subplot(131)
# plt.plot(np.array(theta_path_pgd)[:,0],np.array(theta_path_pgd)[:,1],"r-s",label='PGD')
# # plt.subplot(132)
# plt.plot(np.array(theta_path_sgd)[:,0],np.array(theta_path_sgd)[:,1],"g-+",label='SGD')
# # plt.subplot(133)
# plt.plot(np.array(theta_path_mgd)[:,0],np.array(theta_path_mgd)[:,1],"b-o",label='MGD')
# # plt.axis([2,4,3,5])
# plt.show()

# #多项式回归
m = 100
X = 6*np.random.rand(m,1)-3
X = np.array(X).reshape(100,1)
Y = 0.5*X**2+np.random.randn(m,1)
Y = np.array(Y).reshape(100,1)
# plt.plot(X,Y,"r.")
#
# poly_features = PolynomialFeatures(degree = 2,include_bias=False)
# X_poly = poly_features.fit_transform(X)
# # print(X_poly[0])
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly,Y)
# print(lin_reg.coef_)
# print(lin_reg.intercept_)
#
# X_new = np.linspace(-3,3,100).reshape(100,1)
# X_new_poly = poly_features.fit_transform(X_new)
# y_new = lin_reg.predict(X_new_poly)
# plt.plot(X_new,y_new,'--',label='prediction')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(12,6))
# plt.plot(X,Y,'y.')
# for style,width,degree in (('g--',1,10),('b--',1,2),('r--',1,1)):
#     poly_features = PolynomialFeatures(degree=degree, include_bias=False)
#     std = StandardScaler()
#     lin_reg = LinearRegression()
#     polynomial_reg = Pipeline([('poly_features',poly_features),
#               ('StandardScaler',std),
#               ('lin_reg',lin_reg)])
#     polynomial_reg.fit(X,Y)
#     y_new_2 = polynomial_reg.predict(X_new)
#     plt.plot(X_new,y_new_2,style,linewidth=width,label="degree:"+str(degree))
# plt.legend()
# plt.show()

# # #数据样本数量对结果的影响
# def plot_learning_curves(model,X,Y):
#     X_TR,X_VAL,Y_TR,Y_VAL = train_test_split(X,Y,test_size=0.2,random_state=0)
#     train_errors,val_errors = [],[]
#     for m in range(1,len(X_TR)):
#         model.fit(X_TR[:m],Y_TR[:m])
#         y_train_predict = model.predict(X_TR[:m])
#         y_val_predict = model.predict(X_VAL)
#         train_errors.append(mean_squared_error(Y_TR[:m],y_train_predict))
#         val_errors.append(mean_squared_error(Y_VAL,y_val_predict))
#     plt.plot(np.sqrt(train_errors),'r--',linewidth=3,label="train errors")
#     plt.plot(np.sqrt(val_errors),'b--',linewidth=3,label="val errors")
#     plt.legend()
#     plt.show()
# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg,X,Y)

# #多项式回归的过拟合风险
# polynomial_reg = Pipeline([('poly_features',PolynomialFeatures(degree = 25,include_bias=False)),
#                ('lin_reg',LinearRegression())])
# plot_learning_curves(polynomial_reg,X,Y)

# #正则化:岭回归与lasso,alpha(惩罚力度)越大，曲线越平稳，lasso是平均值
# #J(theta)=MSE(theta)+alpha*1/2*sum(theta**2) 最终损失函数 岭回归
# X_new = np.linspace(-3,3,100)
# X_new = np.array(X_new).reshape(100,1)
# def plot_model(model_class,polynomial,alphas):
#     for alpha,style in zip(alphas,('b-','g-','r-')):
#         model = model_class(alpha)
#         if(polynomial):
#             model = Pipeline([('poly_features', PolynomialFeatures(degree=10,include_bias=False)),
#                            ('StandardScaler',StandardScaler()),
#                            ('lin_reg',model)])
#         model.fit(X,Y)
#         Y_new = model.predict(X_new)
#         lw = 2 if alpha > 0 else 1
#         plt.plot(X_new,Y_new,style,linewidth=lw,label='alpha={}'.format(alpha))
#     plt.plot(X,Y,'b.')
#     plt.legend()
#
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plot_model(Ridge,polynomial=False,alphas=(0,10,100))
# plt.subplot(122)
# plot_model(Ridge,polynomial=True,alphas=(0,0.1,1))
# plt.show()
