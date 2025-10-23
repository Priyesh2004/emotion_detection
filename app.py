import streamlit as st
import joblib as jb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------
# 🔹 Load Model and Assets (cached)
# -------------------------------
@st.cache_resource
def load_model_and_assets():
    model = tf.keras.models.load_model("emotion_detection_model.h5")
    tokenizer = jb.load("tokenizer.jb")
    label_encoder = jb.load("label_encoder.jb")
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_and_assets()


# -------------------------------
# 🔹 Text Preprocessing Function
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove links
    text = re.sub(r'@\w+', '', text)      # remove mentions
    text = re.sub(r'#\w+', '', text)      # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# 🔹 Prediction Function
# -------------------------------
@st.cache_data
def predictor(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


# -------------------------------
# 🔹 Streamlit UI
# -------------------------------
st.title("😊 Emotion Detection from Text")
st.write("Enter a sentence to analyze its emotion")

user_input = st.text_area("Enter text:")

if st.button("Analyze Emotion"):
    if user_input.strip():
        with st.spinner("Analyzing emotion..."):
            prediction = predictor(user_input)
        st.success(f"Predicted Emotion: **{prediction}**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
