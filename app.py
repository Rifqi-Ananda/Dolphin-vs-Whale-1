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
st.title("🧠 Image Classification & Object Detection App")
st.markdown("""
Aplikasi ini memiliki dua fitur utama:
- **Deteksi Objek (YOLO)** untuk mendeteksi objek di dalam gambar.
- **Klasifikasi Gambar** untuk mengidentifikasi jenis **penyakit mata** berdasarkan citra retina.
""")

# Pilih mode
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# Upload gambar
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# Jika ada gambar diupload
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    # --- MODE YOLO ---
    if menu == "Deteksi Objek (YOLO)":
        st.write("🔍 Mendeteksi objek menggunakan model YOLO...")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="📸 Hasil Deteksi Objek", use_container_width=True)

    # --- MODE KLASIFIKASI GAMBAR ---
    elif menu == "Klasifikasi Gambar":
        st.write("🧠 Mengklasifikasikan gambar...")

        # ==========================
        # Preprocessing Aman
        # ==========================
        target_size = classifier.input_shape[1:3]  # otomatis ambil ukuran model
        if classifier.input_shape[-1] == 1:
            img = img.convert("L")  # grayscale
        else:
            img = img.convert("RGB")

        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # Label kelas
        classes = ["Cataract 👁️", "Diabetic Retinopathy 🩸", "Glaucoma 👓", "Normal ✅"]
        predicted_label = classes[class_index]

        # Hasil klasifikasi
        st.markdown(f"## 🩺 Hasil Klasifikasi: **{predicted_label}**")
        st.markdown(f"### 🔢 Probabilitas: `{probability:.2f}`")

        # Tambahkan deskripsi penyakit
        descriptions = {
            "Cataract 👁️": "Terjadi ketika lensa mata menjadi keruh, mengganggu penglihatan.",
            "Diabetic Retinopathy 🩸": "Kerusakan pada retina akibat komplikasi diabetes yang memengaruhi pembuluh darah.",
            "Glaucoma 👓": "Tekanan tinggi dalam bola mata yang dapat merusak saraf optik.",
            "Normal ✅": "Tidak terdeteksi adanya kelainan pada citra retina."
        }

        st.info(descriptions[predicted_label])

        # Tambahkan animasi/gif sesuai hasil
        gif_links = {
            "Cataract 👁️": "https://media.giphy.com/media/l0MYC0LajbaPoEADu/giphy.gif",
            "Diabetic Retinopathy 🩸": "https://media.giphy.com/media/26n6WywJyh39n1pBu/giphy.gif",
            "Glaucoma 👓": "https://media.giphy.com/media/3o6nUQ0KI0n5g7sXBe/giphy.gif",
            "Normal ✅": "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif"
        }

        st.image(gif_links[predicted_label], caption="Animasi Terkait")

else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #89f7fe, #66a6ff);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

