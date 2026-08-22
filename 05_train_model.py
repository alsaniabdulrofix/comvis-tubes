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
# 2. VALIDASI LABEL 3 KELAS
# =====================================================
print("=" * 50)
print("MEMVALIDASI DATASET 3 KELAS...")
print("=" * 50)

y_train_mapped = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
y_test_mapped = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

expected_labels = np.arange(3)
if not np.array_equal(np.unique(y_train_mapped), expected_labels):
    raise ValueError(
        "Label training harus 0=Cricket, 1=Boxing, 2=Tennis. "
        f"Ditemukan: {np.unique(y_train_mapped)}"
    )
if not np.array_equal(np.unique(y_test_mapped), expected_labels):
    raise ValueError(
        "Label test harus 0=Cricket, 1=Boxing, 2=Tennis. "
        f"Ditemukan: {np.unique(y_test_mapped)}"
    )

train_indices = np.arange(len(y_train_mapped))
test_indices = np.arange(len(y_test_mapped))

print("Label: 0=Cricket, 1=Boxing, 2=Tennis")
print(f"Data train: {len(train_indices)} | Data test: {len(test_indices)}")

# =====================================================
# 3. SPLIT VALIDATION & MEMORY MAPPING
# =====================================================
print("Menyiapkan pipeline Memory Mapping...")
X_train_mmap = np.load(os.path.join(DATASET_DIR, "X_train.npy"), mmap_mode='r')
X_test_mmap = np.load(os.path.join(DATASET_DIR, "X_test.npy"), mmap_mode='r')

idx_train, idx_val, y_train_final, y_val_final = train_test_split(
    train_indices, y_train_mapped, test_size=0.2, random_state=42, stratify=y_train_mapped
)

def data_generator(indices, y_data, X_data_mmap):
    for i, idx in enumerate(indices):
        yield X_data_mmap[idx], y_data[i]

def test_generator():
    for i, idx in enumerate(test_indices):
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
