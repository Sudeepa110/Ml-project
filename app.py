import os
import gdown
import streamlit as st
from keras.models import load_model
from PIL import Image
import pandas as pd
import numpy as np

MODEL_PATH = "traffic_sign_model.h5"
# Direct download URL for Google Drive (make sure your file is shared publicly)
url = "https://drive.google.com/uc?id=1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"

def download_model_file():
    """Download the model file and verify it's a valid HDF5 file."""
    st.write("Downloading model file...")
    # Re-download the file (overwrite if exists)
    gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)
    
    # Check if the file exists
    if not os.path.exists(MODEL_PATH):
        st.error("Model file was not downloaded.")
        return False

    # Verify file header (HDF5 files should start with: b'\x89HDF\r\n\x1a\n')
    try:
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(8)
        st.write(f"File header: {header}")
        if header != b'\x89HDF\r\n\x1a\n':
            st.error("Downloaded file is not a valid HDF5 file. It may be an HTML error page or corrupted.")
            return False
    except Exception as e:
        st.error(f"Error reading model file: {e}")
        return False

    st.write("Model file verified as a valid HDF5 file.")
    return True

def preprocess_image(image):
    """Resize to 32x32, convert to RGB if needed, normalize pixel values, and add a batch dimension."""
    image = image.resize((32, 32))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

def predict_traffic_sign(image):
    """Download the model, load it, preprocess the image, and perform a prediction."""
    # Download and verify model file on each prediction
    if not download_model_file():
        st.error("Failed to download or verify the model file.")
        return None, None

    # Load the model (this will run on each prediction click)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    processed_image = preprocess_image(image)
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
