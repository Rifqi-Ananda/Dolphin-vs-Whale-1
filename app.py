import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# CUSTOM STYLE (THEME)
# ==========================
st.markdown("""
<style>
/* Background gradient */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #a8edea, #fed6e3);
}

/* Hapus background header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Glow effect untuk judul */
h1 {
    text-align: center;
    font-size: 60px;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    text-shadow: 0 0 15px rgba(0, 114, 255, 0.5);
}

/* Gaya teks deskripsi */
p {
    text-align: center;
    font-size: 20px;
    color: #222;
}

/* === TAB GLOW STYLE === */
div[data-baseweb="tab"] {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: white !important;
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    border-radius: 10px;
    padding: 10px 20px;
    margin: 5px;
    box-shadow: 0px 0px 20px rgba(0, 150, 255, 0.6);
    transition: all 0.3s ease-in-out;
}

/* Hover dan aktif glowing */
div[data-baseweb="tab"]:hover {
    box-shadow: 0px 0px 25px rgba(0, 255, 255, 0.9);
    transform: scale(1.05);
}

div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    box-shadow: 0px 0px 25px rgba(0, 255, 255, 0.9);
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
<h1>🐬 AI Vision Lab 🧠</h1>
<p>Pilih fitur di bawah untuk mendeteksi <b>dolphin/whale</b> atau <b>menganalisis penyakit mata</b>.</p>
""", unsafe_allow_html=True)

# ==========================
# NAVIGASI TAB
# ==========================
tab1, tab2 = st.tabs(["🌊 Deteksi Dolphin vs Whale", "🩺 Klasifikasi Penyakit Mata"])

# ===================================================
# TAB 1: DETEKSI DOLPHIN vs WHALE
# ===================================================
with tab1:
    st.subheader("🐋 Deteksi Objek Menggunakan YOLOv8")

    uploaded_file = st.file_uploader("📤 Unggah Gambar (Dolphin/Whale)", type=["jpg", "jpeg", "png"], key="detection")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔍 Mendeteksi objek menggunakan model YOLO..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📸 Hasil Deteksi Objek", use_container_width=True)

    else:
        st.info("Silakan unggah gambar lumba-lumba atau paus untuk mendeteksi objek 🐬🐋")

# ===================================================
# TAB 2: KLASIFIKASI PENYAKIT MATA
# ===================================================
with tab2:
    st.subheader("🧠 Klasifikasi Penyakit Mata Menggunakan CNN")

    uploaded_eye = st.file_uploader("📤 Unggah Citra Retina Mata", type=["jpg", "jpeg", "png"], key="classification")

    if uploaded_eye is not None:
        img = Image.open(uploaded_eye)
        st.image(img, caption="👁️ Gambar Retina yang Diupload", use_container_width=True)
        st.write("🧩 Mengklasifikasikan gambar...")

        # --- PREPROCESSING ---
        target_size = classifier.input_shape[1:3]
        if classifier.input_shape[-1] == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # --- PREDIKSI ---
        with st.spinner("🩺 Menganalisis citra retina..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            probability = np.max(prediction)

        # --- LABEL DAN HASIL ---
        classes = ["Cataract 👁️", "Diabetic Retinopathy 🩸", "Glaucoma 👓", "Normal ✅"]
        predicted_label = classes[class_index]

        st.success(f"### 🩺 Hasil Klasifikasi: **{predicted_label}**")
        st.markdown(f"**Probabilitas:** `{probability:.2f}`")

        descriptions = {
            "Cataract 👁️": "Lensa mata menjadi keruh dan mengganggu penglihatan.",
            "Diabetic Retinopathy 🩸": "Kerusakan retina akibat komplikasi diabetes.",
            "Glaucoma 👓": "Tekanan tinggi dalam bola mata merusak saraf optik.",
            "Normal ✅": "Tidak terdeteksi adanya kelainan pada citra retina."
        }

        st.info(descriptions[predicted_label])

        # --- ANIMASI ---
        gif_links = {
            "Cataract 👁️": "https://media.giphy.com/media/l0MYC0LajbaPoEADu/giphy.gif",
            "Diabetic Retinopathy 🩸": "https://media.giphy.com/media/26n6WywJyh39n1pBu/giphy.gif",
            "Glaucoma 👓": "https://media.giphy.com/media/3o6nUQ0KI0n5g7sXBe/giphy.gif",
            "Normal ✅": "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif"
        }

        st.image(gif_links[predicted_label], caption="Animasi Terkait", use_container_width=True)

    else:
        st.info("Silakan unggah citra retina untuk analisis penyakit mata 👁️")
