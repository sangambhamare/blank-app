import streamlit as st
from PIL import Image
import openai

# Set your OpenAI API key here
openai.api_key = "your-openai-api-key"

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

    # Call the OpenAI API to generate a caption
    response = openai.Completion.create(
        engine="gpt-4",  # Use the GPT-4 or GPT-3.5 engine
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Display the generated caption
    caption = response.choices[0].text.strip()
    st.write(f"Generated Caption: {caption}")
