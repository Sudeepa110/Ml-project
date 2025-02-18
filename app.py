import os
import gdown
from keras.models import load_model
from PIL import Image
import pandas as pd
import streamlit as st
import numpy as np
import h5py  # For checking HDF5 validity

MODEL_PATH = "traffic_sign_model.h5"
# Convert the share link to a direct download link if needed:
url = "https://drive.google.com/uc?id=1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"

def download_model_file():
    # Check if file exists and has a reasonable size (e.g., more than 100KB)
    if not os.path.exists(MODEL_PATH) or os.stat(MODEL_PATH).st_size < 100 * 1024:
        st.write("Downloading model file...")
        gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)
    else:
        st.write("Model file already exists.")
        
    # Check file size after download
    file_size = os.stat(MODEL_PATH).st_size
    st.write(f"Model file size: {file_size / 1024:.2f} KB")
    
    # Try opening with h5py to verify it is a valid HDF5 file
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            st.write("Model file verified as a valid HDF5 file.")
    except Exception as e:
        st.error(f"Model file appears corrupted or is not a valid HDF5 file: {e}")
        raise

# Download (or verify) the model file first
download_model_file()

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

def preprocess_image(image):
    """
    Preprocess the input image to match the model's input shape:
      - Resize to 32x32
      - Ensure 3 color channels (RGB)
      - Normalize pixel values
    """
    image = image.resize((32, 32))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize
    processed_image = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return processed_image

def predict_traffic_sign(image):
    """
    Predict the traffic sign class and confidence.
    """
    processed_image = preprocess_image(image)
    model = load_traffic_sign_model()  # Load model on demand (cached)
    if model is None:
        st.error("Model is not loaded.")
        return None, None
    predictions = model.predict(processed_image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_id, confidence

# Load the CSV containing sign names
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
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        class_id, confidence = predict_traffic_sign(image)
        if class_id is not None and sign_names is not None:
            try:
                sign_name = sign_names[sign_names["ClassId"] == class_id]["SignName"].values[0]
            except IndexError:
                sign_name = "Unknown"
            st.write(f"### Predicted Traffic Sign: **{sign_name}**")
            st.write(f"### Confidence: **{confidence * 100:.2f}%**")
