import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    layout="centered"
)

model = tf.keras.models.load_model("potato_model.h5")

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

disease_info = {
    "Early Blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "prevention": "Use fungicides, remove infected leaves, and practice crop rotation."
    },
    "Late Blight": {
        "cause": "Disease caused by Phytophthora infestans.",
        "prevention": "Ensure proper drainage, avoid overwatering, and use resistant varieties."
    },
    "Healthy": {
        "cause": "No disease detected.",
        "prevention": "Continue maintaining good agricultural practices."
    }
}

st.title("Potato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        img = image.resize((256, 256))

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        confidence = np.max(prediction) * 100

        st.subheader("Prediction Result")

        st.write(f"Disease: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Disease Information")

        st.write(
            f"Cause: {disease_info[predicted_class]['cause']}"
        )

        st.write(
            f"Prevention: {disease_info[predicted_class]['prevention']}"
        )