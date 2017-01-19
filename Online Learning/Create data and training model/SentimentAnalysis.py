# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:16:50 2016

@author: Administrator
"""


import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.linear_model import LogisticRegression
#读取数据
df=pd.read_csv('.\movie_data.csv')

def preprocessor(text):
    '''
    1.去掉HTML标签
    2.找出表情并存在emoticons
    3.去掉所以非字母的字符，并连接上表情
    '''
    text=re.sub('<[^>]*>','',text)#去掉HTML标签。
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)#找出表情。
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-','')#去掉非字符，字母转换为小写，加上表情。
    return text

df['review']=df['review'].apply(preprocessor)#所以review用precessor

def tokenizer():
    return text.split()
    
porter=PorterStemmer()           
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

from nltk.corpus import stopwords
stop=stopwords.words('english')

#%%
x_train=df.loc[:25000-1,'review'].values
y_train=df.loc[:25000-1,'sentiment'].values

x_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values

#%%
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(lowercase=False)
param_grid=[{'vect_ngram_range':[(1,1)],
'vect_stop_words':[stop,None],
'vect_tokenizer':['tokenizer','tokenizer_porter'],
'clf_penalty':['l1','l2'],
'clf_C':[1.0,10.0,100.0]},
{'vect_ngram_range':[(1,1)],
'vect_stop_words':[stop,None],
'vect_tokenizer':['tokenizer','tokenizer_porter'],
'clf_penalty':['l1','l2'],
'clf_C':[1.0,10.0,100.0],
'vect_use_idf':[False],
'vect_norm':[None]}]

lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,
                         scoring='accuracy',
                         cv=5,verbose=1,
                         n_jobs=-1)
gs_lr_tfidf.fit(x_train,y_train)

#%%
from sklearn.externals import joblib
joblib.dump(gs_lr_tfidf,'train_model.m')





      