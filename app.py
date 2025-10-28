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
st.title("🐬🐋 Dolphin vs Whale Detection App")

st.markdown("""
Aplikasi ini memungkinkan kamu untuk:
- **Mendeteksi objek** pada gambar menggunakan model YOLO.
- **Mengklasifikasikan** menggunakan model klasifikasi berbasis TensorFlow.
""")

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi"]
)

uploaded_file = st.file_uploader("Unggah Gambar ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🌊 Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek dengan YOLO
        st.write("🔍 **Mendeteksi objek di dalam gambar...**")
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
        st.image(result_img, caption="📸 Hasil Deteksi Objek", use_container_width=True)

    elif menu == "Klasifikasi Penyakit Mata":
        # ==========================
        # Preprocessing
        # ==========================
        img_resized = img.resize((224, 224))  # sesuaikan dengan input model
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

        # ==========================
        # Prediksi
        # ==========================
        st.write("🧠 **Menganalisis gambar untuk mendeteksi penyakit mata...**")
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # ==========================
        # Label Kelas
        # ==========================
        classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
        predicted_label = classes[class_index]

        # ==========================
        # Tampilkan Hasil
        # ==========================
        st.subheader("📊 Hasil Klasifikasi Penyakit Mata")
        st.markdown(f"**Prediksi:** `{predicted_label}`")
        st.markdown(f"**Probabilitas:** `{probability:.2f}`")

        if predicted_label == "Cataract":
            st.warning("⚠️ **Cataract** terdeteksi — terdapat kekeruhan pada lensa mata yang dapat menyebabkan penglihatan kabur.")
        elif predicted_label == "Diabetic Retinopathy":
            st.error("🚨 **Diabetic Retinopathy** terdeteksi — komplikasi akibat diabetes yang dapat merusak retina.")
        elif predicted_label == "Glaucoma":
            st.warning("⚠️ **Glaucoma** terdeteksi — peningkatan tekanan intraokular yang dapat merusak saraf optik.")
        else:
            st.success("✅ **Normal** — tidak terdeteksi tanda-tanda penyakit mata yang signifikan.")
