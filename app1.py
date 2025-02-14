import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Streamlit header
st.header('Image Classification Model')

# Load the pre-trained model
model = load_model('F:/Image_Classification/Image_classify.keras')

# Data categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Image dimensions
img_height = 180
img_width = 180

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Display file name
    st.write("Uploaded file:", uploaded_file.name)

    # Load and preprocess the image
    image_load = Image.open(uploaded_file).resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make a prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Display the image
    st.image(image_load, caption="Uploaded Image", width=200)

    # Display the prediction and accuracy
    st.write('Veg/Fruit in image is:', data_cat[np.argmax(score)])
    st.write('With accuracy of:', round(float(np.max(score) * 100), 2), '%')
