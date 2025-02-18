import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# Define model path and Google Drive URL (using the direct download link format)
MODEL_PATH = "traffic_sign_model.h5"
 url = "https://drive.google.com/uc?export=download&id=1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"


# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()
else:
    st.write(f"Model file exists: {MODEL_PATH}")

# Use Streamlit's caching to load the model once and keep it available
@st.cache_resource
def load_traffic_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_traffic_model()

# Load the sign names from the CSV file
try:
    sign_names = pd.read_csv("signname.csv")
except Exception as e:
    st.error(f"Error loading signname.csv: {e}")
    st.stop()

# Preprocess the uploaded image to match model's expected input
def preprocess_image(image):
    """
    Preprocess the input image:
    - Resize to (32, 32)
    - Convert to RGB if not already
    - Normalize pixel values to [0, 1]
    - Add a batch dimension (1, 32, 32, 3)
    """
    image = image.resize((32, 32))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

# Predict the traffic sign class and confidence
def predict_traffic_sign(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_id, confidence

# Streamlit App UI
st.title("Traffic Sign Classifier")
st.write("Upload a traffic sign image, and the model will classify it.")

# File uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")

    # Predict button
    if st.button("Predict"):
        try:
            with st.spinner("Predicting..."):
                class_id, confidence = predict_traffic_sign(image)
            # Lookup the sign name from signnames.csv based on ClassId
            sign_name_series = sign_names[sign_names["ClassId"] == class_id]["SignName"]
            if sign_name_series.empty:
                st.error(f"Class ID {class_id} not found in signname.csv")
            else:
                sign_name = sign_name_series.values[0]
                st.write(f"### Predicted Traffic Sign: **{sign_name}**")
                st.write(f"### Confidence: **{confidence * 100:.2f}%**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
