# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 21:44:15 2017

@author: Administrator
"""

import sqlite3
import os
conn=sqlite3.connect('review.sqlite')
c=conn.cursor()
c.execute('CREATE TABLE review_db (review TEXT,sentiment INTEGER, date TEXT)')
example1='I love this movie'
c.execute("INSERT INTO review_db(review,sentiment, date) VALUES(?,?,DATETIME('now'))", (example1,1))
example2='I dislike this movie'
c.execute("INSERT INTO review_db(review,sentiment, date) VALUES(?,?,DATETIME('now'))",(example2,0))
conn.commit()
conn.close()