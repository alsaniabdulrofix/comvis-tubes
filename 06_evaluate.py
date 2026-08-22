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
y_test_mapped = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

print(f"Jumlah data test  : {len(y_test_mapped)}")
print(f"Kelas unik        : {np.unique(y_test_mapped)}")

# Generator untuk menghemat RAM saat prediksi
def test_generator():
    for x in X_test_mmap:
        yield x

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
target_names = ['CricketShot', 'Punch', 'TennisSwing']

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