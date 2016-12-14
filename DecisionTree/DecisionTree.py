# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:21:54 2016

@author: Administrator
"""

from sklearn.tree import DecisionTreeClassifier
from my_tools import plot_decision_regions,create_data
import numpy as np
from matplotlib import pyplot as plt

x_train,x_test,y_train,y_test=create_data()
tree=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree.fit(x_train,y_train)
x_combined=np.vstack((x_train,x_test))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(x_combined,y_combined,tree,test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.title('entropy')
plt.show()

#%%
tree=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0)
tree.fit(x_train,y_train)
plot_decision_regions(x_combined,y_combined,tree,test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.title('gini')
plt.show()