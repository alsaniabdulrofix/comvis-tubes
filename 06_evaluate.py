import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# =====================================================
# 1. KONFIGURASI
# =====================================================
DATASET_DIR = "dataset_numpy"
MODEL_PATH = "model/action_recognition.keras"
BATCH_SIZE = 4

print("=" * 50)
print("MEMUAT MODEL DAN DATA TEST...")
print("=" * 50)

# Load model yang sudah dilatih
model = tf.keras.models.load_model(MODEL_PATH)

# =====================================================
# 2. LOAD & FILTER DATA TEST (3 KELAS)
# =====================================================
X_test_mmap = np.load(os.path.join(DATASET_DIR, "X_test.npy"), mmap_mode='r')
y_test_raw = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

# Filter hanya untuk kelas: 0 (CricketShot), 2 (Punch), 4 (TennisSwing)
test_valid_mask = (y_test_raw == 0) | (y_test_raw == 2) | (y_test_raw == 4)
test_indices = np.where(test_valid_mask)[0]

y_test_filtered = y_test_raw[test_indices]

# Ubah label menjadi 0, 1, 2
y_test_mapped = np.zeros_like(y_test_filtered)
y_test_mapped[y_test_filtered == 0] = 0  # Cricket
y_test_mapped[y_test_filtered == 2] = 1  # Boxing
y_test_mapped[y_test_filtered == 4] = 2  # Tennis

# Generator untuk menghemat RAM saat prediksi
def test_generator():
    for idx in test_indices:
        yield X_test_mmap[idx]

output_signature = tf.TensorSpec(shape=(20, 224, 224, 3), dtype=tf.float32)

test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 3. PREDIKSI DATA TEST
# =====================================================
print("\nMelakukan prediksi pada data Test...")
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# =====================================================
# 4. CLASSIFICATION REPORT
# =====================================================
target_names = ['Cricket', 'Boxing', 'Tennis']

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
print("=" * 50)
report = classification_report(y_test_mapped, y_pred, target_names=target_names)
print(report)

# =====================================================
# 5. CONFUSION MATRIX
# =====================================================
print("\n" + "=" * 50)
print("MEMBUAT CONFUSION MATRIX...")
print("=" * 50)

cm = confusion_matrix(y_test_mapped, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation='vertical')
plt.title('Confusion Matrix - 3 Kelas')
plt.tight_layout()

# Simpan gambar
plt.savefig('confusion_matrix.png', dpi=300)
print("Grafik Confusion Matrix berhasil disimpan sebagai 'confusion_matrix.png'")

print("\n" + "=" * 50)
print("TAHAP 6 SELESAI")
print("=" * 50)