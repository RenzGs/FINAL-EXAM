import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os


model_path = 'best_cnn_model.h5'
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=14_Sb6gtncRs7rr4cWTfpdvF_5ppkEdbc'
    gdown.download(url, model_path, quiet=False)  # corrected line here


model = tf.keras.models.load_model(model_path)


class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']


st.title("Weather Image Classifier")


uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)


    st.write(f"Prediction: **{predicted_class}** with **{confidence*100:.2f}% confidence**.")

