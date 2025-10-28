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
# 🔹 Custom CSS for Styling
# -------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size:2.4rem;
        font-weight:600;
        color:#090979;
        letter-spacing:1px;
    }
    .subtitle {
        font-size:1.2rem;
        color:#24243e;
        margin-bottom:16px;
    }
    .result-card {
        background:#f4f6fa;
        padding:1rem 1.5rem;
        border-radius:15px;
        font-size:1.1rem;
        border-left:5px solid #090979;
        margin-top:1rem;
        font-weight:500;
        color:#2d3142;
    }
             /* Animated gradient background for whole app */
    .stApp {
        background: linear-gradient(120deg, #e0c3fc, #8ec5fc, #a9c9ff);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }
    /* Optional: Softer effect for main content */
    .block-container {
        background: rgba(255,255,255,0.83);
        border-radius: 16px;
        padding: 2rem 2rem !important;
        box-shadow: 0 8px 32px rgba(32,56,112,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

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
# Header
st.markdown("<div class='main-title'>Emotion Detection from Text</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🧠 Enter any sentence below and find out what emotion it expresses!</div>", unsafe_allow_html=True)

# Text Input
user_input = st.text_area("Type something to analyze the emotion:", height=120, help="e.g. I'm feeling great today!")

# Button and Display
if st.button("🔍 Analyze Emotion"):
    if user_input.strip():
        with st.spinner("Analyzing emotion..."):
            prediction = predictor(user_input)
        st.markdown(f"<div class='result-card'>Predicted Emotion: <span style='color:#090979'>{prediction}</span></div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("<small>Created with ❤️ using Streamlit</small>", unsafe_allow_html=True)
