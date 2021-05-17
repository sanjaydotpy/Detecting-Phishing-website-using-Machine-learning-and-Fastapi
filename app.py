
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from profanity_check import predict_prob
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("C:\\Users\\vvssa\\C:\Users\vvssa\phishing  project for sarath\\phishing.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    def print_sentiment_scores(tweets):
        vadersenti = analyser.polarity_scores(tweets)
        return pd.Series([vadersenti['pos'], vadersenti['neg'], vadersenti['neu'], vadersenti['compound']])

    messages=[]
    for i in request.form.values():
        messages.extend(i.split('--'))
    feed_data=[len(messages)] #number of conversation
    feed_data.append(0.75) #percent of conversations started by the author
    avgsec=0
    meantime=0
    for i in messages:
        avgsec+=int(i[i.index('[')+1:i.index(']')])
    meantime=avgsec
    avgsec/=len(messages)
    diffsec=0
    t=0
    for i in messages[1:]:
        diffsec+=int(i[i.index('[')+1:i.index(']')])-t
        t=int(i[i.index('[')+1:i.index(']')])
    diffsec/=len(messages)
    feed_data.append(diffsec) #difference between two preceding lines in seconds
    feed_data.append(len(messages)) #number of messages sent
    nol=0
    for i in messages:
        if len(i)>1:nol+=1
    nol/=len(messages)
    feed_data.append(nol) #average percent of lines in conversation
    qp=0
    for i in messages:
        qp+=i.count('?')/len(i)
    qp/=len(messages)
    feed_data.append(nol-qp) #average percent of characters in conversation
    autchar=0
    qmark=0
    aqmark=0
    acount=0
    for i in messages:
        qmark+=i.count('?')
        if 'pedo' in i:
            acount+=1
            aqmark+=i.count('?')
            autchar+=len(i)-5
    totqmark=qmark
    qmark/=len(messages)
    feed_data.append(autchar) #number of characters sent by the author
    feed_data.append(meantime) #mean time of messages sent
    feed_data.append(2) #number of unique contacted authors
    feed_data.append(0) #avg number of unique authors interacted with per conversation
    feed_data.append(2) #total unique authors and unique per chat difference
    feed_data.append(30) #conversation num and total unique authors difference
    feed_data.append(qmark) #average question marks per conversations
    feed_data.append(totqmark) #total question marks
    feed_data.append(aqmark) #total author question marks
    feed_data.append(aqmark/acount) #avg author question marks
    feed_data.append(totqmark-aqmark) #author and conversation quetsion mark differnece




    pos=neg=neu=comp=0
    wp=wn=wnu=wc=0
    for i in messages:
        if i[0]=='p':
            tpos,tneg,tneu,tcomp=print_sentiment_scores(i[5:])
            pos+=tpos
            wp+=int(tpos*len(i))
            neg+=tneg
            wn+=int(tneg*len(i))
            neu+=tneu
            comp+=tcomp

    feed_data.append(neg) #author total negative in author conv
    feed_data.append(neu) #author total neutral in author conv 
    feed_data.append(pos) #author total positive in author conv
    feed_data.append(comp) #authortotal compound in author conv
    feed_data.append(wp) #pos word count author
    feed_data.append(wn) #neg word count author


    pcount=0
    for i in messages:
        if i[0]=='p':pcount+=predict_prob([i[5:]])[0]*len(i)

    feed_data.append(pcount) #prof word count author

    final_features = [np.array(feed_data)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output<0.5:return render_template('index.html', prediction_text='He is not suspected as a pedophile')
    else:return render_template('index.html', prediction_text='He is suspected as a pedophile')
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)







