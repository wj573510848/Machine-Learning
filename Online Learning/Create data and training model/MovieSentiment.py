# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 22:18:56 2017

@author: Administrator
"""
'''
大数据下，在线学习算法。Online algorithm and out of core learning
1.
 使用随机梯度下降方法，
'''

import numpy as np
import os
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind

stop=stopwords.words('english')
def tokenizer(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-','')
    tokenized=[w for w in text.split() if w not in stop]
    return tokenized
#加载数据，，每次输出一个文档（one review）
def steam_docs(path):
    with open(path,'r') as csv:
        next(csv)#跳过第一行标题。
        for line in csv:
            text,label=line[:-3],int(line[-2])
            yield text,label
#获得指定大小的数据集。
def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y
#使用HashingVectorizer向量化数据。
vect=HashingVectorizer(decode_error='ignore',
                       n_features=2**21,
                       preprocessor=None,
                       tokenizer=tokenizer)
clf=SGDClassifier(loss='log',random_state=1,n_iter=1,n_jobs=-1)
doc_stream=steam_docs('./movie_data.csv')
pbar=pyprind.ProgBar(45)
classes=np.array([0,1])
for _ in range(45):
    x_train,y_train=get_minibatch(doc_stream,size=1000)
    if not x_train:
        break
    x_train=vect.transform(x_train)
    clf.partial_fit(x_train,y_train,classes=classes)
    pbar.update()
#%% 测试
x_test,y_test=get_minibatch(doc_stream,size=5000)
x_test=vect.transform(x_test)
print('Accuracy: %.3f' % clf.score(x_test,y_test))
clf=clf.partial_fit(x_test,y_test)

#%%
import pickle
import os
dest=os.path.join('movieclassifier','pkl_objects')        
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'))
pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'))
         
            
    

