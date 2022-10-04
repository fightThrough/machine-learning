#关于sklearn工具的学习
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score#交叉验证测准确率
from sklearn.base import clone#将一个模型复制成多份
from sklearn.model_selection import StratifiedKFold#将数据集分成K层
from sklearn.model_selection import cross_val_predict#也是交叉验证
from sklearn.metrics import confusion_matrix#混淆矩阵
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
# mnist = fetch_mldata("MNIST original")
# mnist
iris = load_iris()
X=iris["data"]
Y=iris["target"]
shuffle_index_train = np.random.permutation(150)
X,Y = X[shuffle_index_train],Y[shuffle_index_train]#需要把数据集进行洗牌操作
X_train,X_test,Y_train,Y_test = X[:130],X[130:],Y[:130],Y[130:]
Y_train_5 = (Y_train==1)
Y_test_5 = (Y_test==1)#更改标签
SGD_CLF = SGDClassifier()
SGD_CLF.fit(X_train,Y_train_5)
# print(sum(SGD_CLF.predict(X_test) == Y_test_5))
# print(SGD_CLF.predict(X_test))
# print(Y_test_5)
# model_score = cross_val_score(SGD_CLF,X_train,Y_train_5,cv=3,scoring='accuracy')
# print(model_score)
#将训练集分为训练集与测试集
#如下是交叉验证
skfolds = StratifiedKFold(n_splits=3)
for train_index,test_index in skfolds.split(X_train,Y_train_5):
    clone_clf = clone(SGD_CLF)
    X_train_folds = X_train[train_index]
    Y_train_folds = Y_train_5[train_index]
    X_test_folds = X_train[test_index]
    Y_test_folds = Y_train[test_index]
    clone_clf.fit(X_train_folds,Y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == Y_test_folds)
    print(n_correct/len(y_pred))
y_predict = cross_val_predict(SGD_CLF,X_train,Y_train_5,cv=3)
print(confusion_matrix(Y_train_5,y_predict))#详见API文档
print(SGD_CLF.decision_function([X_train[25]]))#预测得分
Y_scores=SGD_CLF.decision_function(X_train)
print(Y_scores)
t=10#阈值越小，recall越大，精度越低
Y_pred = (Y_scores>t)
print(Y_pred)
#sklearn不允许自己设置阈值，调用函数
precisions,recalls,thresholds=precision_recall_curve(Y_train_5,Y_scores)
print(thresholds.shape)#一共129个阈值，以及对应每个阈值精度，recall
plt.figure(figsize=(8,4))
plt.plot(thresholds,precisions[:-1],"b",label="precisions")
plt.plot(thresholds,recalls[:-1],"g",label="recalls")
plt.xlim([-150,150])
plt.ylim([0,1])
plt.show()
#也可以画准确率与召回率的曲线
plt.figure()
plt.plot(precisions[:-1],recalls[:-1])
plt.show()

print(roc_auc_score(Y_train_5,Y_scores))#ROC的AUC
#ROC曲线
#X-FP   Y-TP  理想AUC(曲线下面积)趋近于1

# print(1)