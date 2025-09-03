import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import os
from pathlib import Path

try:
    from streamlit_image_select import image_select  # optional, nicer selection
    HAS_IMAGE_SELECT = True
except Exception:
    HAS_IMAGE_SELECT = False

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnv2_pro_best.keras")

model = load_model()

# Class labels
CLASS_NAMES = [
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Target_Spot',
 'Tomato___healthy'
 ]

st.title("Plant Disease Detection App")
st.subheader("Upload or choose an example.")

uploaded_file = st.file_uploader("Upload a Tomato leaf image", type=["jpg", "jpeg", "png"])


st.subheader("Try with Example Images")


APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
example_files = []
if EXAMPLES_DIR.exists():
    example_files = [
        f for f in sorted(os.listdir(EXAMPLES_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

selected_example = None

if example_files:
    if HAS_IMAGE_SELECT:
        images = []
        captions = []
        for file in example_files:
            img_path = EXAMPLES_DIR / file
            try:
                images.append(Image.open(img_path))
                captions.append(file)
            except Exception:
                continue
        if images:
            choice = image_select(
                label="Click an example to select",
                images=images,
                captions=captions,
                use_container_width=True,
                return_value="index",
                key="example_gallery",
            )
            if isinstance(choice, int) and 0 <= choice < len(example_files):
                selected_example = str(EXAMPLES_DIR / example_files[choice])
    else:
        cols = st.columns(4)
        for i, file in enumerate(example_files):
            col = cols[i % 4]
            with col:
                img_path = EXAMPLES_DIR / file
                st.image(str(img_path), caption=file, use_container_width=True)
                if st.button("Use", key=f"use_{file}"):
                    selected_example = str(img_path)
else:
    st.info("No example images found in `app/examples`. Add some .jpg/.png files to that folder.")


image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif selected_example is not None:
    image = Image.open(selected_example).convert("RGB")

# -----------------------------
# Prediction
# -----------------------------
if image is not None:
    st.image(image, caption="Selected Leaf Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.success(f"Prediction: **{CLASS_NAMES[predicted_class]}** ({confidence*100:.2f}% confidence)")
