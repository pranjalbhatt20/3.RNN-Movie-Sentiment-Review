import pandas as pd
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
word_index_reverse={v:k for k,v in word_index.items()}

model=load_model('./saved model/rnn_imdb.h5')

# Function to decode reviews 
def decode_review(encoded_review):
    return ' '.join([word_index_reverse.get(i-3,'?') for i in encoded_review])

# function to preprocess user input 
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# prediction function 
def predict_sentiment(review):
    preprocess_review=preprocess_text(review)
    prediction=model.predict(preprocess_review)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

# Stream lit app 
import streamlit as sc
sc.title("IMDB Moview review sentiment analysis")
sc.write("Write review below")

# taking user input 
user_input=sc.text_area("Moview Review")

if sc.button('Classify'):
    sentiment,score=predict_sentiment(user_input)
    sc.write(sentiment)
else:
    sc.write("Please enter a movie review")