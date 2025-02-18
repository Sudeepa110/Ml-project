import os
from keras.models import load_model
import gdown
from PIL import Image
import pandas as pd
import streamlit as st
import numpy as np

MODEL_PATH = "traffic_sign_model.h5"
url = "https://drive.google.com/file/d/1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"

# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
else:
    st.write(f"Model file exists: {MODEL_PATH}")

# Cache the model loading so that it's loaded only once
@st.cache_resource
def load_traffic_sign_model():
    try:
        model = load_model(MODEL_PATH)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the input image to match the model's input shape.
    - Resize the image to 32x32
    - Ensure it has 3 color channels (RGB)
    - Normalize the pixel values
    """
    # Resize to (32, 32)
    image = image.resize((32, 32))

    # Convert to RGB if the image is grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert the image to a numpy array and normalize
    image_array = np.array(image) / 255.0  # Normalize to range [0, 1]

    # Add a batch dimension (1, 32, 32, 3)
    processed_image = np.expand_dims(image_array, axis=0)

    return processed_image

# Predict the class of the image using the model loaded on demand
def predict_traffic_sign(image):
    """
    Predict the traffic sign class and confidence.
    - image: PIL.Image object
    Returns:
    - class_id: Predicted class
    - confidence: Prediction confidence
    """
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Load the model when needed (cached)
    model = load_traffic_sign_model()
    if model is None:
        st.error("Model is not loaded.")
        return None, None

    # Predict using the model
    predictions = model.predict(processed_image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_id, confidence

# Load the CSV containing sign names
sign_names = pd.read_csv("signname.csv")  # Ensure this file is in the same directory

# Streamlit app
st.title("Traffic Sign Classifier")
st.write("Upload a traffic sign image, and the model will classify it.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        class_id, confidence = predict_traffic_sign(image)
        if class_id is not None:
            # Get the corresponding sign name from signnames.csv
            sign_name = sign_names[sign_names["ClassId"] == class_id]["SignName"].values[0]
            st.write(f"### Predicted Traffic Sign: **{sign_name}**")
            st.write(f"### Confidence: **{confidence * 100:.2f}%**")
