# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:10:40 2016

@author: Administrator
"""

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

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
