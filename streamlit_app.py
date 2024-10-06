import streamlit as st
import io
from google.cloud import vision

# Replace with your Google Cloud project ID
project_id = "vibrant-crawler-437800-c4"

st.title("Image Labeler (Google Cloud Vision API)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    st.timer("Creating a Vision Client...")

    # Create a Vision client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    image = vision.Image(content=uploaded_file.getvalue())

    # Perform label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Display the detected labels
    st.write("## Detected Labels:")
    for label in labels:
        st.write(f"Label: {label.description}, Score: {label.score}")
