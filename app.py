import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load model and tokenizer
model = load_model('model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Data cleaning function
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    return " ".join(text)

def predict_hate_speech(text):
    cleaned_text = clean_text(text)
    st.write(f"Cleaned text: {cleaned_text}")  # Debugging information
    seq = tokenizer.texts_to_sequences([cleaned_text])
    st.write(f"Tokenized sequence: {seq}")  # Debugging information
    if not seq or len(seq[0]) == 0:
        st.error("Error: The input text could not be tokenized. Please check the input.")
        return None
    padded = pad_sequences(seq, maxlen=300)
    pred = model.predict(padded)
    return "Hate Speech" if pred >= 0.5 else "Not Hate Speech"

st.title("Hate Speech Classification")
st.write("Enter text or upload a text file to classify.")

input_type = st.radio("Choose input type", ("Text", "File"))

if input_type == "Text":
    user_input = st.text_area("Enter text here")
    if st.button("Classify"):
        result = predict_hate_speech(user_input)
        if result:
            st.write(f"Result: {result}")

elif input_type == "File":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode('utf-8')
        result = predict_hate_speech(file_text)
        if result:
            st.write(f"Result: {result}")
