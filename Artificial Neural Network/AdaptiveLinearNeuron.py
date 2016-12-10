# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 21:24:45 2016

@author: Administrator
"""
import numpy as np
from matplotlib import pyplot as plt

class AdalineGD(object):
    def __init__(self,eta=0.01,n_iter=50):
        self.eta=eta
        self.n_iter=n_iter
    
    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]
    
    def activation(self,x):
        return self.net_input(x)
    
    def predict(self,x):
        return np.where(self.activation(x)>=0,1,-1)
    
    def fit(self,x,y):
        self.w_=np.zeros(1+x.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            output=self.net_input(x)
            errors=(y-output)
            self.w_[1:]+=self.eta*x.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=sum(errors**2)/2.0
            self.cost_.append(cost)
        return self
        
import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values
x_std=np.copy(x)
x_std[:,0]=(x[:,0]-x[:,0].mean())/x[:,0].std()
x_std[:,1]=(x[:,1]-x[:,1].mean())/x[:,1].std()
#%% 
#不同学习率对梯度下降的影响，
fig,ax=plt.subplots(1,2,figsize=(8,4))
ada1=AdalineGD(n_iter=10,eta=0.01).fit(x,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline-Learning rate 0.01')

ada2=AdalineGD(n_iter=10,eta=0.0001).fit(x,y)
ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline-Learning rate 0.0001')
plt.show()
#%%
#标准化后的结果。Standardization。
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
        
ada=AdalineGD(n_iter=50,eta=0.01)
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

        