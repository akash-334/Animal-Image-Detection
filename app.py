# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:48:46 2024

@author: 91898
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\91898\Detection.h5')

# Set up Streamlit app
st.title("Cat and Dog Classifier")

st.write("""
         This app classifies uploaded images as either **Cat** or **Dog**.
         Upload an image, and the model will predict its category.
         """)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    st.write("Classifying...")
    img = image.resize((150, 150))  # Resize to match model input size
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict with the model
    prediction = model.predict(img_array)
    class_name = "Dog" if prediction[0] > 0.5 else "Cat"
    confidence = prediction[0][0] if class_name == "Dog" else 1 - prediction[0][0]
    
    st.write(f"Prediction: **{class_name}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")


