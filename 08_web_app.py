import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import tempfile

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="HaYOLOh", page_icon="🎥", layout="centered")

# ==========================================
# 2. CUSTOM CSS (Memaksa Warna Teks)
# ==========================================
st.markdown("""
    <style>
    /* Sembunyikan header bawaan Streamlit */
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}

    /* Background Gradasi Biru Tua ke Putih */
    .stApp {
        background: linear-gradient(to bottom, #0a192f 0%, #172a45 35%, #e6e9f0 70%, #ffffff 100%);
    }
    
    /* Logo HaYOLOh di pojok kiri atas */
    .hayoloh-logo {
        position: fixed;
        top: 15px;
        left: 25px;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 2px;
        text-shadow: 0px 2px 8px rgba(0,0,0,0.5);
        z-index: 99999;
    }

    /* Memaksa Kotak Utama Menjadi Putih Solid */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.3) !important;
        border: none !important;
    }

    /* MEMAKSA SEMUA TEKS DI DALAM KOTAK JADI HITAM (Solusi Teks Hilang) */
    [data-testid="stVerticalBlockBorderWrapper"] * {
        color: #121212 !important;
    }

    /* Mengembalikan warna teks tombol ke putih */
    .stButton button * {
        color: #ffffff !important;
    }

    /* Desain Tombol Prediksi */
    .stButton > button {
        background: linear-gradient(to right, #1e88e5, #1565c0) !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(30,136,229,0.4) !important;
        margin-top: 15px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30,136,229,0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER UTAMA (LOGO)
# ==========================================
st.markdown('<div class="hayoloh-logo">HaYOLOh</div>', unsafe_allow_html=True)

# ==========================================
# 4. FUNGSI INTI AI
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/action_recognition.keras")

model = load_model()
CLASSES = ["Cricket", "Boxing", "Tennis"]
IMG_SIZE = 224
MAX_FRAMES = 20

def extract_frames(video_path, max_frames=20, img_size=224):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        cap.release()
        return None
        
    frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    frames = []
    current = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        current += 1
        
    cap.release()
    
    if len(frames) == 0:
        return None
        
    while len(frames) < max_frames:
        frames.append(frames[-1])
        
    return np.array(frames, dtype=np.float32)

# ==========================================
# 5. KOTAK KONTEN UTAMA
# ==========================================
with st.container(border=True):
    # Mengunci teks judul area unggah dengan warna hitam
    st.markdown('<h3 style="color: #121212; margin-top: -10px; margin-bottom: 20px;">📤 Area Unggah Video</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Format file yang didukung: MP4, AVI", type=['mp4', 'avi'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Analisis Video Sekarang 🚀", use_container_width=True):
            with st.spinner('Memproses pola gerakan...'):
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                frames = extract_frames(tfile.name, MAX_FRAMES, IMG_SIZE)
                
                if frames is not None:
                    input_data = np.expand_dims(frames, axis=0)
                    predictions = model.predict(input_data)[0]
                    predicted_class_idx = np.argmax(predictions)
                    predicted_label = CLASSES[predicted_class_idx]
                    confidence = predictions[predicted_class_idx] * 100
                    
                    st.write("---")
                    
                    # Membuat Kotak Hasil Analisis Secara Manual dengan HTML agar warnanya tidak bentrok
                    st.markdown(f"""
                    <div style="background-color: #e8f5e9; border: 2px solid #81c784; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                        <h4 style="color: #2e7d32 !important; margin: 0; text-align: center;">✅ Aksi Terdeteksi: <strong>{predicted_label}</strong> ({confidence:.2f}%)</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mengunci warna judul probabilitas
                    st.markdown('<p style="color: #121212 !important; font-weight: bold; font-size: 1.1rem;">Tingkat Kepercayaan Sistem (Probabilitas):</p>', unsafe_allow_html=True)
                    
                    for i, label in enumerate(CLASSES):
                        # Mengunci warna teks persentase
                        st.markdown(f'<p style="color: #333333 !important; margin-bottom: -15px; font-weight: 600;">{label} : {predictions[i]*100:.2f}%</p>', unsafe_allow_html=True)
                        st.progress(float(predictions[i]))
                else:
                    st.error("Sistem gagal membaca susunan frame pada video.")