import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
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

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2 , stratify=Y,random_state=2)
model = LogisticRegression()
model.fit(X_train,Y_train)

#accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of the training data :", training_data_accuracy)

#accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of the training data :", test_data_accuracy)

#Prediction

X_tn, X_tt, Y_tn, Y_tt = train_test_split(news_dataset['content'],news_dataset['label'],test_size = 0.2 , stratify=Y,random_state=2)
tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_ngram.fit(X_tn,Y_tn)
predicted_LogR_ngram = logR_pipeline_ngram.predict(X_tt)
np.mean(predicted_LogR_ngram == Y_tt)


model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram,open(model_file,'wb'))
load_model = pickle.load(open('final_model.sav', 'rb'))
var = input("Enter the news : ")
prediction = load_model.predict([var])
prob = load_model.predict_proba([var])
print(prediction,prob)


if(prediction[0]==0) :
	print('The news is Real')
else :
	print('The news is Fake')





