# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 21:50:10 2016

@author: Administrator
"""

import numpy as np
from matplotlib import pyplot as plt

#训练数组x为[n_samples,n_features].
#训练数组先乘以权值，x*weight，然后用激活函数分类。
#>=0，为1，否则为-1.
class perceptron_simple(object):
    def __init__(self,iteration=100,theta=0.1):#定义循环次数，学习率，学习率范围0`1。
        self.n_iter=iteration
        self.theta=theta
    
    def net_input(self,x):
        #self.w_为权值。
        return np.dot(x,self.w_[1:])+self.w_[0]
    
    def predict(self,x):
        return np.where(self.net_input(x)>=0,1,-1)
        
    def fit(self,x,y):
        self.w_=np.zeros(1+x.shape[1])
        self.errors_=[]
        for _ in range(self.n_iter):
            error=0
            for xi,yi in zip(x,y):
                update=self.theta*(yi-self.predict(xi))
                self.w_[0]=update+self.w_[0]
                self.w_[1:]+=update*xi
                error+=int(update!=0)
            self.errors_.append(error)
        return self
    
    def plot(self,x,y):
        plt.plot(x[y==-1][:,0],x[y==-1][:,1],'ro')
        plt.plot(x[y==1][:,0],x[y==1][:,1],'b+')
        x_show=[min(x[:,0])-1,max(x[:,0])+1]
        x_show=np.array(x_show)
        y_show=self.w_[0]/-self.w_[2]+self.w_[1]*x_show/-self.w_[2]
        plt.plot(x_show,y_show)
        plt.xlim(x_show[0],x_show[1])
        plt.ylim(y_show[0],y_show[1])
        plt.show()
        

#%%
#用Iris dataset验证该算法。
import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
#5. class: 
      #-- Iris Setosa
      #-- Iris Versicolour
      #-- Iris Virginica
#%%
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values
plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#%%
ppn=perceptron_simple(iteration=10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='v')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
#%%
#decision boundaries for 2D datasets
from matplotlib.colors import ListedColormap

def plot_decision_regions(x,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')#cyan 青色
    cmap=ListedColormap(colors[:len(np.unique(y))])
    
    x1_min,x1_max=x[:,0].min()-1,x[:,0].max()+1
    x2_min,x2_max=x[:,1].min()-1,x[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x[y==cl,0],x[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
plot_decision_regions(x,y,ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()




























