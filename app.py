import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    return " ".join(tokens)

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transform_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else: 
        st.header("Not spam")