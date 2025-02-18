import os
import time
import gdown
import streamlit as st
from keras.models import load_model
from PIL import Image
import pandas as pd
import numpy as np

MODEL_PATH = "traffic_sign_model.h5"
# Direct download URL for Google Drive (ensure your file is shared publicly)
url = "https://drive.google.com/uc?id=1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"

def download_model_if_needed():
    """Download the model if it doesn't exist or if it's older than a certain interval."""
    # For example, re-download if the file is older than 1 hour (3600 seconds)
    reload_interval = 3600
    if not os.path.exists(MODEL_PATH) or (time.time() - os.path.getmtime(MODEL_PATH)) > reload_interval:
        st.write("Downloading model file...")
        gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)
    else:
        st.write("Using cached model file.")

@st.cache_resource(show_spinner=False)
def load_cached_model():
    """Download (if needed) and load the model (this function is cached)."""
    download_model_if_needed()
    try:
        model = load_model(MODEL_PATH)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Resize image to 32x32, convert to RGB if needed, normalize pixel values, and add a batch dimension."""
    image = image.resize((32, 32))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

def predict_traffic_sign(image):
    """Preprocess the image, load the cached model, and perform a prediction."""
    processed_image = preprocess_image(image)
    model = load_cached_model()
    if model is None:
        st.error("Model is not loaded.")
        return None, None
    predictions = model.predict(processed_image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_id, confidence

# Load CSV with sign names
try:
    sign_names = pd.read_csv("signname.csv")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    sign_names = None

# --- Streamlit App ---
st.title("Traffic Sign Classifier")
st.write("Upload a traffic sign image, and the model will classify it.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Predict"):
        class_id, confidence = predict_traffic_sign(image)
        if class_id is not None and sign_names is not None:
            try:
                sign_name = sign_names[sign_names["ClassId"] == class_id]["SignName"].values[0]
            except IndexError:
                sign_name = "Unknown"
            st.write(f"### Predicted Traffic Sign: **{sign_name}**")
            st.write(f"### Confidence: **{confidence * 100:.2f}%**")
