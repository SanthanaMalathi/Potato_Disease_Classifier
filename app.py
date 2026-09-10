import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🥔",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

MODEL_PATH = Path("potato_disease_model.keras")
CLASS_NAMES_PATH = Path("class_names.json")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return class_names


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(model, image, class_names):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image to model input size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert image to NumPy array
    img = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img, verbose=0)[0]

    # Find predicted class
    predicted_index = np.argmax(predictions)

    # Get confidence
    confidence = float(predictions[predicted_index])

    # Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain", confidence, predictions

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence, predictions


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_class_name(class_name):

    if class_name == "Potato___Early_blight":
        return "Early Blight"

    elif class_name == "Potato___Late_blight":
        return "Late Blight"

    elif class_name == "Potato___healthy":
        return "Healthy"

    return class_name


# ============================================================
# DISEASE INFORMATION
# ============================================================

def get_disease_information(class_name):

    information = {

        "Potato___Early_blight": {
            "title": "Potato Early Blight",
            "description": (
                "The model predicts that the potato leaf may show "
                "symptoms associated with Early Blight."
            )
        },

        "Potato___Late_blight": {
            "title": "Potato Late Blight",
            "description": (
                "The model predicts that the potato leaf may show "
                "symptoms associated with Late Blight."
            )
        },

        "Potato___healthy": {
            "title": "Healthy Potato Leaf",
            "description": (
                "The model predicts that the uploaded potato leaf "
                "appears healthy."
            )
        }
    }

    return information.get(
        class_name,
        {
            "title": "Unknown",
            "description": "No additional information available."
        }
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🥔 Potato Leaf Disease Classifier")

st.write(
    """
    Upload an image of a potato leaf and the deep learning model
    will classify it as **Early Blight**, **Late Blight**, or
    **Healthy**.
    """
)

st.info(
    "This application is a machine-learning demonstration and "
    "should not replace expert agricultural diagnosis."
)


# ============================================================
# LOAD MODEL AND CLASS NAMES
# ============================================================

try:

    model = load_model()
    class_names = load_class_names()

except Exception as e:

    st.error("Unable to load the model or class mapping.")

    st.exception(e)

    st.stop()


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write("**Architecture:** EfficientNetB0")
    st.write("**Input Size:** 224 × 224 pixels")
    st.write("**Number of Classes:** 3")
    st.write("**Confidence Threshold:** 60%")

    st.write("**Classes:**")

    for class_name in class_names:
        st.write(f"- {format_class_name(class_name)}")


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Potato Leaf",
        use_container_width=True
    )

    # Prediction button
    if st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence, predictions = predict_image(
                model,
                image,
                class_names
            )

        st.divider()

        # ====================================================
        # UNCERTAIN PREDICTION
        # ====================================================

        if predicted_class == "Uncertain":

            st.warning(
                f"⚠️ Prediction is uncertain.\n\n"
                f"Model confidence: {confidence:.2%}"
            )

            st.write(
                "Please upload a clearer potato leaf image "
                "with the leaf occupying most of the frame."
            )

        # ====================================================
        # CONFIDENT PREDICTION
        # ====================================================

        else:

            formatted_class = format_class_name(predicted_class)

            st.subheader("Prediction")

            st.success(
                f"### {formatted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

            # =================================================
            # DISEASE INFORMATION
            # =================================================

            info = get_disease_information(predicted_class)

            st.write(f"**{info['title']}**")

            st.write(info["description"])

            # =================================================
            # PROBABILITY DISTRIBUTION
            # =================================================

            st.subheader("Prediction Probabilities")

            probability_data = {}

            for class_name, probability in zip(
                class_names,
                predictions
            ):

                formatted_name = format_class_name(class_name)

                probability_data[formatted_name] = float(probability)

            st.bar_chart(probability_data)

            # =================================================
            # DETAILED PROBABILITIES
            # =================================================

            for class_name, probability in probability_data.items():

                st.write(
                    f"**{class_name}:** {probability:.2%}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built using TensorFlow, EfficientNetB0 and Streamlit."
)