import streamlit as st

st.title("Image Uploader")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  # Display the uploaded image
  st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

  # You can add code here to process the uploaded image 
  # (e.g., object detection, image manipulation)

  # ...
