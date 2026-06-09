import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "models/character_model.h5"
    )
    return model

model = load_model()

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.prediction-card {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(
        90deg,
        #4facfe,
        #00f2fe
    );
    color: white;
}

.metric-card {
    padding: 15px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO SECTION
# =====================================

st.markdown("""
<div style="
padding:30px;
border-radius:20px;
text-align:center;
background:linear-gradient(
90deg,
#667eea,
#764ba2
);
">

<h1 style="color:white;">
✍️ AI Handwritten Character Recognition
</h1>

<h3 style="color:white;">
Deep Learning Based Character Recognition System
</h3>

<p style="color:white;font-size:18px;">
Upload or draw a handwritten character and let
Artificial Intelligence predict the Character instantly.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================
# PROJECT STATS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        "99%"
    )

with col2:
    st.metric(
        "Dataset",
        "MNIST"
    )

with col3:
    st.metric(
        "Classes",
        "10 Characters"
    )

st.markdown("---")
# =====================================

# INPUT METHODS

# =====================================

tab1, tab2 = st.tabs([
"✍️ Draw Character",
"📤 Upload Image"
])

image_for_prediction = None

# =====================================

# DRAW DIGIT TAB

# =====================================

with tab1:

 st.subheader(
    "Draw a Character (A-Z)"
)

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:

    image_for_prediction = Image.fromarray(
        canvas_result.image_data.astype("uint8")
    ).convert("RGB")


# =====================================

# IMAGE UPLOAD TAB

# =====================================

with tab2:

 st.subheader(
    "Upload a Handwritten Digit"
)

uploaded_file = st.file_uploader(
    "Choose Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

if uploaded_file is not None:

    image_for_prediction = Image.open(
        uploaded_file
    )

    st.image(
        image_for_prediction,
        width=250
    )

# =====================================

# PREPROCESS IMAGE

# =====================================

def preprocess_image(image):

    image = image.convert("L")

    image = ImageOps.invert(image)

    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    image = np.array(image)

    image = image / 255.0

    image = image.reshape(
        1,
        28,
        28,
        1
    )

    return image

st.markdown("---")

predict_btn = st.button(
"🚀 Predict character"
)
## ==========================================
# PREDICTION SECTION
# ==========================================

if predict_btn and image_for_prediction is not None:

    processed_image = preprocess_image(
        image_for_prediction
    )

    st.image(
        processed_image[0, :, :, 0],
        caption="Image Sent To AI",
        width=200
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    # =====================================
    # CHARACTER PREDICTION
    # =====================================

    characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    predicted_class = np.argmax(prediction)

    predicted_character = characters[predicted_class]

    confidence = float(
        np.max(prediction)
    ) * 100

    st.markdown("---")

    st.success(
        f"✅ Predicted Character: {predicted_character}"
    )

    # =====================================
    # RESULT CARD
    # =====================================


    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )
        # =====================================
        # PROBABILITY CHART
        # =====================================

    st.divider()

    st.subheader(
            "📊 Character Prediction Probabilities"
        )

    probabilities = prediction[0] * 100

    chart_df = pd.DataFrame({
            "Character": characters,
            "Probability": probabilities
        })

    fig = px.bar(
        chart_df,
            x="Character",
            y="Probability",
            title="A-Z Character Prediction"
        )

    st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================
    # TOP 5 PREDICTIONS
    # =====================================

    st.subheader(
        "🏆 Top 5 Predictions"
    )

    top_indices = np.argsort(
        prediction[0]
    )[::-1][:5]

    top_df = pd.DataFrame({
        "Character": [
            characters[i]
            for i in top_indices
        ],
        "Confidence (%)": [
            round(
                float(prediction[0][i]) * 100,
                2
            )
            for i in top_indices
        ]
    })

    st.dataframe(
        top_df,
        use_container_width=True
    )

    # =====================================
    # PROJECT INFORMATION
    # =====================================

    st.markdown("---")

    st.markdown("""

## 🤖 About This Project

This application uses:

✅ Convolutional Neural Networks (CNN)

✅ Deep Learning

✅ TensorFlow & Keras

✅ A-Z Handwritten Character Dataset

✅ Handwritten Character Recognition

### Workflow

✍️ Draw / Upload Character

⬇️

🧠 CNN Model

⬇️

🎯 Character Prediction

⬇️

📊 Confidence Analysis

Developed for the CodeAlpha Machine Learning Internship.

""")

    st.markdown("---")

    st.caption(
        "AI Handwritten Character Recognition | CNN + TensorFlow + Deep Learning"
    )