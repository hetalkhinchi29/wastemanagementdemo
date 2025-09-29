# app.py (Full-screen Attractive UI)
import streamlit as st
#import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

# --- Custom CSS (modern full-screen design) ---
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #d4fc79, #96e6a1);
            padding: 0;
            margin: 0;
        }
        .upload-box {
            background: rgba(255,255,255,0.85);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .result-card {
            background: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.25);
            text-align: center;
            margin-top: 20px;
        }
        .bin-box {
            font-size: 26px;
            font-weight: bold;
            padding: 15px;
            border-radius: 15px;
            margin: 20px auto;
            color: white;
            width: 250px;
        }
        .confidence {
            font-size: 20px;
            margin-top: 15px;
            font-weight: 600;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 14px;
            color: #333;
            opacity: 0.8;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
MODEL_PATH = "waste_mobilenetv2.h5"
#model = tf.keras.models.load_model(MODEL_PATH)

# --- Load class names ---
CLASS_NAMES_FILE = "class_names.json"
if Path(CLASS_NAMES_FILE).exists():
    class_names = json.loads(Path(CLASS_NAMES_FILE).read_text())
else:
    class_names = ['biodegradable', 'hazardous', 'recyclable']
    st.warning(f"{CLASS_NAMES_FILE} not found — using fallback {class_names}")

# --- Bin mapping ---
bin_map = {
    'biodegradable': {
        'bin_name': 'Green Bin',
        'emoji': '🟢',
        'hex': '#28a745',
        'desc': 'Organic waste like food scraps, leaves, etc.'
    },
    'recyclable': {
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclables: paper, plastic, glass, metals.'
    },
    'hazardous': {
        'bin_name': 'Red Bin',
        'emoji': '🔴',
        'hex': '#dc3545',
        'desc': 'Hazardous waste: batteries, chemicals, e-waste.'
    }
}

# --- Preprocessing ---
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    #img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --- Header ---
st.markdown("<h1 style='text-align:center; color:#0b3d02;'>♻️ Smart Waste Classification System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#1c6006;'>Detect waste type and get the right bin recommendation instantly</h3>", unsafe_allow_html=True)

# --- Layout: Full screen with two columns ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload an image of waste", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file:
        # Predict
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        probs = predictions[0]
        top_idx = np.argmax(probs)
        top_label = class_names[top_idx]
        top_conf = float(probs[top_idx] * 100)

        # --- Result Card ---
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown(f"<h2>🏷 Prediction: <b>{top_label.capitalize()}</b></h2>", unsafe_allow_html=True)
        st.progress(int(top_conf))
        st.markdown(f"<div class='confidence'>Confidence: {top_conf:.2f}%</div>", unsafe_allow_html=True)

        bin_info = bin_map.get(top_label.lower())
        if bin_info:
            st.markdown(
                f"<div class='bin-box' style='background:{bin_info['hex']}'>"
                f"{bin_info['emoji']} {bin_info['bin_name']}"
                f"</div>",
                unsafe_allow_html=True
            )
            st.write(bin_info['desc'])
        else:
            st.warning("⚠️ No bin mapping found for this category.")

        # Top-3 predictions
        st.write("🔎 **Confidence Scores:**")
        for i in np.argsort(probs)[::-1][:3]:
            st.write(f"- {class_names[i]}: {probs[i]*100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div class='footer'>🌍 Made with ❤️ for a cleaner planet | Powered by AI</div>", unsafe_allow_html=True)
