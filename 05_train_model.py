import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# =====================================================
# 1. KONFIGURASI UMUM
# =====================================================
DATASET_DIR = "dataset_numpy"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "action_recognition.keras")
BATCH_SIZE = 4
EPOCHS = 30  # Dinaikkan agar model punya waktu menyempurnakan akurasi

# =====================================================
# 2. FILTER & MAPPING 3 KELAS
# =====================================================
print("=" * 50)
print("MENYARING DATASET MENJADI 3 KELAS...")
print("=" * 50)

y_train_full_raw = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
y_test_raw = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

train_valid_mask = (y_train_full_raw == 0) | (y_train_full_raw == 2) | (y_train_full_raw == 4)
test_valid_mask = (y_test_raw == 0) | (y_test_raw == 2) | (y_test_raw == 4)

train_indices_filtered = np.where(train_valid_mask)[0]
test_indices_filtered = np.where(test_valid_mask)[0]

y_train_filtered = y_train_full_raw[train_indices_filtered]
y_test_filtered = y_test_raw[test_indices_filtered]

def remap_labels(y_array):
    y_new = np.zeros_like(y_array)
    y_new[y_array == 0] = 0  # Cricket
    y_new[y_array == 2] = 1  # Boxing
    y_new[y_array == 4] = 2  # Tennis
    return y_new

y_train_mapped = remap_labels(y_train_filtered)
y_test_mapped = remap_labels(y_test_filtered)

# =====================================================
# 3. SPLIT VALIDATION & MEMORY MAPPING
# =====================================================
print("Menyiapkan pipeline Memory Mapping...")
X_train_mmap = np.load(os.path.join(DATASET_DIR, "X_train.npy"), mmap_mode='r')
X_test_mmap = np.load(os.path.join(DATASET_DIR, "X_test.npy"), mmap_mode='r')

idx_train, idx_val, y_train_final, y_val_final = train_test_split(
    train_indices_filtered, y_train_mapped, test_size=0.2, random_state=42, stratify=y_train_mapped
)

def data_generator(indices, y_data, X_data_mmap):
    for i, idx in enumerate(indices):
        yield X_data_mmap[idx], y_data[i]

def test_generator():
    for i, idx in enumerate(test_indices_filtered):
        yield X_test_mmap[idx], y_test_mapped[i]

# =====================================================
# 4. TF.DATA.DATASET
# =====================================================
output_signature = (
    tf.TensorSpec(shape=(20, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(idx_train, y_train_final, X_train_mmap),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(idx_val, y_val_final, X_train_mmap),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 5. ARSITEKTUR CNN + LSTM (PENYEMPURNAAN)
# =====================================================
print("\nMembangun model CNN + LSTM yang Dioptimalkan...")
model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'), input_shape=(20, 224, 224, 3)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(GlobalAveragePooling2D()),
    
    LSTM(64, return_sequences=False),
    Dropout(0.5), # Ditingkatkan untuk mencegah overfitting
    
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 6. CALLBACKS & TRAINING
# =====================================================
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
# Menambahkan rem otomatis jika akurasi melambat
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 50)
print("MEMULAI PROSES TRAINING (VERSI PENYEMPURNAAN)...")
print("=" * 50)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# =====================================================
# 7. EVALUASI AWAL (DATA TEST)
# =====================================================
print("\n" + "=" * 50)
print("MELAKUKAN EVALUASI PADA DATA TEST...")
print("=" * 50)
test_loss, test_acc = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Model berhasil disimpan di: {MODEL_PATH}")

print("\n" + "=" * 50)
print("TAHAP 5 SELESAI")
print("=" * 50)