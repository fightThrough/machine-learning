import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib.image import imread
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# blob_centers = np.array([[0.2,2.3],
#                         [-1.5,2.3],
#                         [-2.8,1.8],
#                         [-2.8,2.8],
#                         [-2.8,1.3]])
# blob_std = np.array([0.4,0.3,0.1,0.1,0.1])
# x,y = make_blobs(n_samples=2000,centers=blob_centers,cluster_std=blob_std,random_state=7)#构造数据
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)
# plt.scatter(x[:,0][y==0],x[:,1][y==0],label='0')
# plt.scatter(x[:,0][y==1],x[:,1][y==1],label='1')
# plt.scatter(x[:,0][y==2],x[:,1][y==2],label='2')
# plt.scatter(x[:,0][y==3],x[:,1][y==3],label='3')
# plt.scatter(x[:,0][y==4],x[:,1][y==4],label='4')
# plt.legend()
# plt.show()
# k = 5
# k_means = KMeans(n_clusters=k,random_state=0)
# y_predict = k_means.fit_predict(x)#得到预测结果,与调用labels属性得到结果一样
# print(y_predict)
# print(k_means.labels_)
# print(k_means.cluster_centers_)
# X_new = [[0,2],[2,3]]
# y_predict_new = k_means.predict(x_new)
# print(y_predict_new)
# def plot_data(X):
#     plt.plot(X[:,0],X[:,1],'k.',markersize=2)
# def plot_centroids(centroids,weights=None,circle_color='w',cross_color='k'):
#     if weights is not None:
#         centroids = centroids[weights > weights.max() / 10]
#     plt.scatter(centroids[:,0],centroids[:,1],marker='o',s=30,linewidths=8,color=circle_color,zorder=10,alpha=0.9)
#     plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=50,linewidths=50,color=cross_color,zorder=11,zlpha=1)
# def plot_decision_boundary(clusters,x,resolution=1000,show_centroids=True,show_xlabels=True,show_ylabels=True):
#     pass#画图
# print(k_means.transform(X_new))#到各个簇之间的距离
# KMeans(n_clusters=5,init='random',n_init=1,max_iter=1,random_state=1)#跑n_init次取最好的一次
# KMeans(n_clusters=5,init='random',n_init=1,max_iter=2,random_state=1)#跑n_init次取最好的一次
# KMeans(n_clusters=5,init='random',n_init=1,max_iter=3,random_state=1)#跑n_init次取最好的一次
#k_means.inertia#评估指标
# for i in [3,4,5,6,7,8,9,10]:
#     k_means = KMeans(n_clusters=i,random_state=0)
#     k_means.fit(x)
#     print(k_means.inertia_)
# k_means_per_k = [KMeans(n_clusters=k).fit(x)for k in range(1,10)]
# inertias = [model.inertia_ for model in k_means_per_k]
# plt.figure(figsize=(12,4))
# plt.plot(range(1,10),inertias,'bo-')
# plt.show()#找拐点选k值比较合适
#轮廓系数si：簇内不相似度 ai   簇间不相似度 bi
from sklearn.metrics import silhouette_score#求轮廓系数
# print(silhouette_score(x,k_means.labels_))
# k_means_per_k = [KMeans(n_clusters=k).fit(x)for k in range(2,10)]
# silhouette = [silhouette_score(x,model.labels_) for model in k_means_per_k]
# print(silhouette)
# plt.figure(figsize=(12,4))
# plt.plot(range(2,10),silhouette,'bo-')
# plt.show()#找拐点选k值比较合适

# x1,y1 = make_blobs(n_samples=1000,centers=((4,-4),(0,0)),random_state=42)
# x1 = x1.dot(np.array([[0.374,0.95],[0.732,0.598]]))
# x2,y2 = make_blobs(n_samples=250,centers=1,random_state=42)
# x2 = x2 + [6, -8]
# x = np.r_[x1,x2]
# plt.plot(x1[:,0],x1[:,1],'b.')
# plt.plot(x2[:,0],x2[:,1],'r.')
# plt.show()
#
# k_means_good = KMeans(n_clusters=3,init=np.array([[-1,2],[0,0.5],[4,1]]),n_init=1,random_state=42)
# k_means_bad  = KMeans(n_clusters=3,n_init=1,random_state=42)
# y_predict_good = k_means_good.fit_predict(x)
# y_predict_bad  = k_means_bad.fit_predict(x)
# # plt.figure(figsize=(10,4))
# print(y_predict_good)
# # print(x.shape)
# for type in np.unique(y_predict_good):
#     plt.scatter(x[:,0][y_predict_good == type],x[:,1][y_predict_good == type])
# plt.show()
# for type in np.unique(y_predict_bad):
#     plt.scatter(x[:,0][y_predict_bad == type],x[:,1][y_predict_bad == type])
# plt.show()

# image = imread('ladybug.png')
# print(image.shape)
# image = image.reshape(-1,3)
# print(image.shape)#每个数据三个特征
# k_means = KMeans(n_clusters=8,random_state=42).fit(image)
# print(k_means.cluster_centers_)
# print(k_means.labels_.shape)
# print(k_means.cluster_centers_[k_means.labels_])
# k_means.cluster_centers_[k_means.labels_].reshape(533, 800, 3)

# segmentd_imgs = []
# n_colors = (10,8,6,4,2)
# for n_cluster in n_colors:
#     k_means = KMeans(n_clusters=n_cluster, random_state=42).fit(image)
#     segmentd_imgs.append(k_means.cluster_centers_[k_means.labels_].reshape(533, 800, 3))
# plt.figure(figsize=(10,5))
# plt.subplot(231)
# image = image.reshape(533, 800, 3)
# plt.imshow(image)
# plt.title('原始图像')
# for index,n_clusters in enumerate(n_colors):
#     plt.subplot(232+index)
#     plt.imshow(segmentd_imgs[index])
#     plt.title('n_colors={}'.format(n_clusters))
# plt.show()


#半监督学习选择哪些个50个参数学习
x_digits,y_digits = load_digits(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x_digits,y_digits,random_state=42)
# print(x_digits.shape)
# print(y_digits.shape)
# print(x_train.shape)
from sklearn.linear_model import LogisticRegression
n_labels = 50
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train[:n_labels],y_train[:n_labels])
print(log_reg.score(x_test,y_test))

k = 50
k_means = KMeans(n_clusters = k, random_state=42)
x_digits_dist = k_means.fit_transform(x_train)
print(x_digits_dist)
print(np.argmin(x_digits_dist,axis=0))#可以找到离每个均值点最近的样本是谁
print(x_digits_dist[23])
x_representative = x_train[np.argmin(x_digits_dist,axis=0)]
log_reg2 = LogisticRegression(random_state=42)
log_reg2.fit(x_representative,y_train[np.argmin(x_digits_dist,axis=0)])
print(log_reg2.score(x_test,y_test))

#标签传播

