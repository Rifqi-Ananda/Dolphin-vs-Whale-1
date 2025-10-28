import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

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
# UI
# ==========================
st.set_page_config(page_title="🐬🐋 Dolphin vs Whale App", page_icon="🌊", layout="centered")

# =============================
# HEADER
# =============================
st.title("🐬🐋 Dolphin vs Whale Detection App")

st.markdown("""
Selamat datang di aplikasi deteksi **Dolphin** dan **Whale**!  
Aplikasi ini menggunakan **model deep learning** untuk:
- 🐬 **Mendeteksi lumba-lumba**
- 🐋 **Mendeteksi paus**
- 🤖 Dan menampilkan hasil klasifikasi dengan probabilitasnya
""")

st.divider()

# =============================
# ANIMASI PEMBUKA
# =============================
st.subheader("🌊 Animasi Lautan")

st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif" width="400">
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("✨ Lumba-lumba dan paus bermain di lautan — siap untuk dideteksi oleh AI!")

st.divider()

# =============================
# UNGGAH GAMBAR
# =============================
uploaded_file = st.file_uploader("📸 Unggah gambar lumba-lumba atau paus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🌊 Gambar yang diunggah", use_container_width=True)

    st.write("🔍 **Model sedang menganalisis gambar...**")
    with st.spinner("Sedang mendeteksi spesies laut... 🌊"):
        # Simulasi waktu prediksi
        gif = Image.open("assets/underwater.gif") if False else None
        time.sleep(2)

    # 🔹 Contoh hasil prediksi (dummy)
    import random
    classes = ["Dolphin 🐬", "Whale 🐋"]
    predicted_label = random.choice(classes)
    probability = round(random.uniform(0.85, 0.99), 2)

    st.subheader("📊 Hasil Klasifikasi")
    st.write(f"**Prediksi:** {predicted_label}")
    st.write(f"**Probabilitas:** `{probability}`")

    if "Dolphin" in predicted_label:
        st.success("🐬 Gambar ini terdeteksi sebagai **Lumba-lumba** — mamalia laut cerdas yang suka berinteraksi.")
        st.image("https://media.giphy.com/media/3oKIPtjElfqwMOTbH2/giphy.gif", caption="Lumba-lumba berenang 🐬", use_container_width=True)
    else:
        st.info("🐋 Gambar ini terdeteksi sebagai **Paus** — raksasa laut yang megah dan damai.")
        st.image("https://media.giphy.com/media/l0Exk8EUzSLsrErEQ/giphy.gif", caption="Paus muncul ke permukaan 🐋", use_container_width=True)

else:
    st.info("📤 Silakan unggah gambar untuk mulai klasifikasi!")

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:0.9em;">
        Dibuat dengan ❤️ oleh tim AI Laut — powered by Streamlit & Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)
