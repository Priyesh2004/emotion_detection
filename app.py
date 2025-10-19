import streamlit as st
import joblib as jb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np

model = tf.keras.models.load_model("emotion_detection_model.h5")

tokenizer = jb.load("tokenizer.jb")
label_encoder = jb.load("label_encoder.jb")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'#\w+','', text)
    text = re.sub(r'[^a-z\s]','',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predictor(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

st.title("Emotion Detection")
st.write("Enter a Sentence to Analyze its emotion")

user_input = st.text_area("Enter text:")
if st.button("Analyze Emotion"):
    if user_input:
        prediction = predictor(user_input)
        st.write(f"Predicted Emotion: {prediction}")
    else:
        st.write("Please enter some texts to analyze")
