import gdown  # Install using `pip install gdown` for Google Drive downloads
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "traffic_sign_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1I5QMX2hgGvIEKcHbqHFZ31R5XjE1Sr5c"  # Replace with the direct download link
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
