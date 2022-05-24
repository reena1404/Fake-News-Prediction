# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:52:56 2022

@author: 91927

"""
from preprocessing import *
from sklearn.pipeline import Pipeline

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
print("Accuracy score of the test data :", test_data_accuracy)


# Saving The Model on the Disc
X_tn, X_tt, Y_tn, Y_tt = train_test_split(news_dataset['content'],news_dataset['label'],test_size = 0.2 , stratify=Y,random_state=2)
tfidf= TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True)
model = Pipeline([
        ('LogR_tfidf',tfidf),
        ('LogR_clf',LogisticRegression())
        ])

model.fit(X_tn,Y_tn)
predicted = model.predict(X_tt)


model_file = 'final_model.sav'
pickle.dump(model,open(model_file,'wb'))