import os
import cv2
import numpy as np
import tensorflow as tf

# =====================================================
# 1. KONFIGURASI
# =====================================================
MODEL_PATH = "model/action_recognition.keras"

# UBAH NAMA FILE INI sesuaikan dengan nama file video kamu!
# Jika video ditaruh di folder utama proyek, tuliskan nama filenya langsung:
VIDEO_PATH = "4.mp4" 

IMG_SIZE = 224
MAX_FRAMES = 20
CLASSES = ["Cricket", "Boxing", "Tennis"]

# =====================================================
# 2. FUNGSI EKSTRAKSI FRAME
# =====================================================
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

    # Padding jika jumlah frame kurang dari 20
    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames, dtype=np.float32)

# =====================================================
# 3. PROSES PREDIKSI
# =====================================================
print("=" * 50)
print("MEMPROSES PREDIKSI VIDEO...")
print("=" * 50)

if not os.path.exists(VIDEO_PATH):
    print(f"Error: File '{VIDEO_PATH}' tidak ditemukan!")
    print("Pastikan nama file dan lokasinya di variabel VIDEO_PATH sudah benar.")
    exit()

print(f"Memuat video: {VIDEO_PATH}")
frames = extract_frames(VIDEO_PATH, MAX_FRAMES, IMG_SIZE)

if frames is None:
    print("Gagal membaca frame dari video!")
    exit()

# Menambahkan dimensi batch: (20, 224, 224, 3) -> (1, 20, 224, 224, 3)
input_data = np.expand_dims(frames, axis=0)

print("Memuat model AI...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Menganalisis gerakan...")
predictions = model.predict(input_data)[0]
predicted_class_idx = np.argmax(predictions)
predicted_label = CLASSES[predicted_class_idx]
confidence = predictions[predicted_class_idx] * 100

# =====================================================
# 4. HASIL PREDIKSI
# =====================================================
print("\n" + "=" * 50)
print("HASIL PREDIKSI AI")
print("=" * 50)
print(f"Aksi Terdeteksi   : {predicted_label}")
print(f"Tingkat Keyakinan : {confidence:.2f}%")
print("\nProbabilitas Per Kelas:")
for idx, label in enumerate(CLASSES):
    print(f"- {label:<8} : {predictions[idx] * 100:.2f}%")

print("=" * 50)