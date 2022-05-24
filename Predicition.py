# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:53:00 2022

@author: 91927
"""

#Prediction
import pickle


def detecting_fake_news(var):    

    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([var])
    #prob = load_model.predict_proba([var])
    if(prediction[0]==0) :
    	prediction = 'Real'
    else :
    	prediction = 'Fake'
    return prediction



