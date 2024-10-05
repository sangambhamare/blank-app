import streamlit as st
from PIL import Image
import openai
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get OpenAI API key from the environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Access the OpenAI API key from Streamlit Secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit UI
st.title("Intelligent Caption Generator")

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the image to bytes for further use
    image_bytes = uploaded_file.read()

    # Generate a prompt to send to OpenAI GPT
    prompt = "Describe this image in a catchy social media caption and suggest some hashtags."

    # Call the OpenAI Chat API to generate a caption using GPT-3.5 or GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo" for models
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.7
    )

    # Display the generated caption
    caption = response['choices'][0]['message']['content'].strip()
    st.write(f"Generated Caption: {caption}")
