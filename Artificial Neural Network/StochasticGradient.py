# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:12:23 2016

@author: Administrator
"""

import numpy as np
from my_tools import plot_decision_regions
from matplotlib import pyplot as plt

class AdalineSGD(object):
    '''ADAptive LInear NEuron classifier.
    
    参数
    ------
    eta：float
    学习率（0-1）
    n_iter:int
    训练集训练次数（epoch）
    
    属性
    ------
    w_:1D-array
    训练后的权值
    errors_:list
    每个epoch后，分类不正确的次数
    shuffle:bool(default:True)
    每个循环随机打乱训练集
    random_state:int（default：None）
    设置随机起始值
    self.cost_:array
    每个循环的平均损失函数值
    
    '''
    def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=None):
        self.eta=eta
        self.n_iter=n_iter
        self.shuffle=shuffle
        self.random_state=random_state
        self.w_initialized=None
        if self.random_state:
            np.random.seed(self.random_state)
        
    def fit(self,x,y):
        '''训练数据集
        参数
        ------
        x:数组，[n_sample,n_features]
        y:数组，[n_sample]
        '''
        
        self._initialize_weights(x.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            if self.shuffle:
                x,y=self._shuffle(x,y)
            cost=[]
            for xi,yi in zip(x,y):
                cost.append(self._update_weights(xi,yi))
            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self,x,y):
        '''新加入数据训练分类器'''
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
            if y.ravel().shape[0]>1:
                for xi,yi in zip(x,y):
                    self._update_weights(xi,yi)
            else:
                self._update_weights(x,y)
            return self
    
    def _shuffle(self,x,y):
        '''随机打乱训练集'''
        r=np.random.permutation(len(y))
        return x[r],y[r]
    
    def _update_weights(self,xi,yi):
        '''更新权值'''
        output=self.activation(xi)
        errors=(yi-output)
        self.w_[1:]+=self.eta*xi.dot(errors)
        self.w_[0]+=self.eta*errors
        cost=0.5*errors**2
        return cost
        
            
            
    def _initialize_weights(self,n_features):
        '''将权值初始化为0'''
        self.w_=np.zeros(n_features+1)
        self.w_initialized=True
        
    def net_input(self,x):
        ''' 计算输入值 '''
        return np.dot(x,self.w_[1:])+self.w_[0]
        
    def activation(self,x):
        '''计算线性激活函数'''
        return self.net_input(x)
        
    def predict(self,x):
        '''计算预测值'''
        return np.where(self.activation(x)>=0.0,1,-1)

import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values
x_std=np.copy(x)
x_std[:,0]=(x[:,0]-x[:,0].mean())/x[:,0].std()
x_std[:,1]=(x[:,1]-x[:,1].mean())/x[:,1].std()

ada=AdalineSGD(n_iter=15,eta=0.01,random_state=1)
ada.fit(x_std,y)
plot_decision_regions(x_std,y,classifier=ada)
plt.title('Adaline-Gradient Descent')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1,1+len(ada.cost_)),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    