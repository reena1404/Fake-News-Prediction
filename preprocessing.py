# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:52:50 2022

@author: 91927
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

import nltk
nltk.download('stopwords')
print(stopwords.words('english'))

#Data Preprocessing

news_dataset = pd.read_csv('train.csv')
print(news_dataset.shape)
print(news_dataset.head())
print(news_dataset.isnull().sum())

#Replacing null value with empty string
news_dataset = news_dataset.fillna('')

#merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title'] 
print(news_dataset['content'])

X = news_dataset.drop(columns='label',axis=1)
Y = news_dataset['label']

print(X)
print(Y)

#Stemming 

port_stem = PorterStemmer()

def stemming(content):
	stemmed_content = re.sub('[^a-zA-Z]',' ', content)
	stemmed_content = stemmed_content.lower()
	stemmed_content = stemmed_content.split()
	stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
	stemmed_content = ' '.join(stemmed_content)
	return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming)
print(news_dataset['content'])

X=news_dataset['content'].values
Y=news_dataset['label'].values

print(X)
print(Y)
print(Y.shape)

# converting textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X)
