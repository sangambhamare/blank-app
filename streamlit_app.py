import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

st.title("Image Feature Extractor")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Load and preprocess the image
    image = cv2.imdecode(uploaded_file.getvalue(), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)

    # Create the ResNet50 model
    model = ResNet50(weights='imagenet', include_top=False)

    # Extract features
    features = model.predict(image[None, ...])

    # Display extracted features
    st.write("## Extracted Features:")
    st.write(features)

    # Get the label with the highest probability
    predicted_class = tf.keras.applications.resnet50.decode_predictions(features)[0][0]
    label = predicted_class[1]
    probability = predicted_class[2]

    # Display predicted label and probability
    st.write("## Predicted Label:")
    st.write(f"Label: {label}, Probability: {probability}")
