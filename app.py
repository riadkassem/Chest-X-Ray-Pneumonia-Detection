import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Force CPU inference
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable mixed precision if any
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# App title
st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and the model will predict whether it is **Normal** or **Pneumonia**.")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.keras")   # change name if needed

model = load_model()
IMG_SIZE = (150, 150)

# -------------------------
# File uploader
# -------------------------
uploaded = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Load + display image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # shape: (1,150,150,3)

    # Prediction button
    if st.button("Predict"):
        pred = model.predict(img_arr, verbose=0)[0][0]

        if pred > 0.5:
            st.error(f"⚠️ Prediction: **Pneumonia** (Confidence: {pred:.2f})")
        else:
            st.success(f"✅ Prediction: **Normal** (Confidence: {1 - pred:.2f})")
