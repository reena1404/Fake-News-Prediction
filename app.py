# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:02:28 2022

@author: 91927
"""

from flask import Flask, request
import Predicition
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)
@app.route('/', methods=['POST'])
def sms():
    resp = MessagingResponse()
    inbMsg = request.values.get('Body')
    pred  = Predicition.detecting_fake_news(inbMsg)

    resp.message(
        f'The news headline you entered is {pred!r}.')
    return str(resp)

if __name__ == '__main__':
    app.run()