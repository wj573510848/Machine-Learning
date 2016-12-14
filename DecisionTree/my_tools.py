# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:10:40 2016

@author: Administrator
"""

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):
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
    if test_idx:
        x_test,y_test=x[test_idx,:],y[test_idx]
        plt.scatter(x_test[:,0],x_test[:,1],c='',alpha=1.0,linewidth=1,marker='o',s=55,label='test set')

def create_data(std=False):
    iris=datasets.load_iris()
    x=iris.data[:,[2,3]]
    y=iris.target
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    if std:
        sc=StandardScaler()
        sc.fit(x_train)
        x_train_std=sc.transform(x_train)
        x_test_std=sc.transform(x_test)
        return x_train_std,x_test_std,y_train,y_test
    else:
        return x_train,x_test,y_train,y_test
    