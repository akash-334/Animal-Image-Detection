import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the TensorFlow Lite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Predict function for TensorFlow Lite model
def predict_tflite(interpreter, img_array):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return prediction[0][0]

# Initialize Streamlit app
st.title("Cat and Dog Classifier")

st.write("""
         This app classifies uploaded images as either **Cat** or **Dog**.
         Upload an image, and the model will predict its category.
         """)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    st.write("Classifying...")
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Load and predict using the TensorFlow Lite model
    interpreter = load_tflite_model(r'C:\Users\91898\Detection.tflite')
    prediction = predict_tflite(interpreter, img_array)
    
    # Interpret the result
    class_name = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if class_name == "Dog" else 1 - prediction
    
    st.write(f"Prediction: **{class_name}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
