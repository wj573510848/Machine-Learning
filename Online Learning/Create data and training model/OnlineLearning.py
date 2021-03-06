# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 21:31:26 2017

@author: Administrator
"""

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
import numpy as np

cur_dir=os.path.dirname(__file__)
stop=pickle.load(open(os.path.join(cur_dir,'movieclassifier','pkl_objects','stopwords.pkl'),'r'))

def tokenizer(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-','')
    tokenized=[w for w in text.split() if w not in stop]
    return tokenized
    
vect=HashingVectorizer(decode_error='ignore',
                       n_features=2**21,
                       preprocessor=None,
                       tokenizer=tokenizer)

clf=pickle.load(open(os.path.join(cur_dir,'movieclassifier','pkl_objects','classifier.pkl'),'r'))

label={0:'negtive',1:'positive'}
example=['I dislike this movie']
x=vect.transform(example)
print('Prediction:%s\nProbability:%.2f%%' % (label[clf.predict(x)[0]],np.max(clf.predict_proba(x))*100))



