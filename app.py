# app.py

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
vectorizer, model = joblib.load('best_model.joblib')

# Text Preprocessing
stemmer = PorterStemmer()
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter news headline/content to predict whether it's real or fake.")

user_input = st.text_area("Enter the News Text Here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vectorized_input = vectorizer.transform([processed])
        result = model.predict(vectorized_input)

        if result[0] == 0:
            st.success("✅ The News is Real.")
        else:
            st.error("❌ The News is Fake.")
