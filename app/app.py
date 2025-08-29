# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "mnv2_pro_best.keras"
CLASS_NAMES = ['Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Target_Spot',
 'Tomato___healthy']

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
normalization_layer = layers.Rescaling(1./255)

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # add batch dim
    img_array = normalization_layer(img_array)
    return img_array

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍅 Tomato Leaf Disease Classifier")
st.write("Upload an image or take a photo to classify the tomato leaf condition.")

option = st.radio("Choose input method:", ["Upload from file", "Use camera"])

if option == "Upload from file":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        st.subheader(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
        st.write(f"Confidence: {confidence:.2f}")

elif option == "Use camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        st.subheader(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
        st.write(f"Confidence: {confidence:.2f}")
