import os
from keras.models import load_model
import gdown

MODEL_PATH = "traffic_sign_model.h5"
url = "https://drive.google.com/file/d/1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"

# Download the model
gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)

# Check if the file exists and is valid
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Model file exists: {MODEL_PATH}")

# Try loading the model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
