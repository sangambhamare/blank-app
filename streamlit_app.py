import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import openai
import os

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to extract labels from image
def extract_labels_from_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    labels = decode_predictions(preds, top=3)[0]
    return [(label[1], label[2]) for label in labels]

# Function to generate caption using OpenAI API
def generate_caption(labels):
    prompt = f"Create a catchy social media caption based on these labels: {', '.join([label[0] for label in labels])}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can change the model if you want
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit UI
st.title("Intelligent Caption Generator")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Extract labels
    labels = extract_labels_from_image(img)
    st.write("Extracted Labels:")
    for label, confidence in labels:
        st.write(f"{label}: {confidence:.2f}")

    # Get OpenAI API key from Streamlit secrets
    openai.api_key = st.secrets["openai"]["api_key"]

    # Generate caption
    caption = generate_caption(labels)
    st.write("Generated Caption:")
    st.write(caption)

