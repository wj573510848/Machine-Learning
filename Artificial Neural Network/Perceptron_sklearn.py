# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 21:53:27 2016

@author: Administrator
"""

'''
训练机器学习算法的一般步骤：
1.选择特征
2.选择矩阵表达
3.选择分类器与优化算法
4.评估模型
5.Tuning the algorithm
'''
#从datasets载入数据
from sklearn import datasets
import numpy as np
iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target

#将数据分割为训练集与测试集
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#标准化
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)

#sklearn的perceptron
from sklearn.linear_model import Perceptron
ppn=Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(x_train_std,y_train)
y_pred=ppn.predict(x_test_std)
print('Misclassified samples: %d' % (y_test!=y_pred).sum())

#metrics的应用
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))

#用decision regions 画图
from my_tools import plot_decision_regions
from matplotlib import pyplot as plt
x_combined_std=np.vstack((x_train_std,x_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(x_combined_std,y_combined,ppn,range(len(x_train_std),len(x_combined_std)))
plt.xlabel('petal length (standardlized)')
plt.ylabel('petal width (standardlized)')
plt.legend(loc='upper left')
plt.show()








