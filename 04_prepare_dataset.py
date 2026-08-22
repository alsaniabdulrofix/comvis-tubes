import os
import cv2
import numpy as np
import pandas as pd

# =====================================
# KONFIGURASI
# =====================================

TRAIN_FOLDER = "Dataset/train"
TEST_FOLDER = "Dataset/test"

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

IMG_SIZE = 224
MAX_FRAMES = 20


# =====================================
# FUNGSI EXTRACT FRAME
# =====================================

def extract_frames(video_path, max_frames=20, img_size=224):

    cap = cv2.VideoCapture(video_path)

    # Mendapatkan jumlah frame video
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Jika video tidak memiliki frame
    if frame_count == 0:
        cap.release()
        return None

    # Menentukan posisi frame yang akan diambil
    frame_indices = np.linspace(
        0,
        frame_count - 1,
        max_frames,
        dtype=int
    )

    frames = []
    current = 0

    # Membaca video
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Jika frame sekarang termasuk frame yang dipilih
        if current in frame_indices:

            # BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize menjadi 224 x 224
            frame = cv2.resize(
                frame,
                (img_size, img_size)
            )

            # Normalisasi pixel 0-255 menjadi 0-1
            frame = frame.astype(
                np.float32
            ) / 255.0

            frames.append(frame)

        current += 1

    cap.release()

    # Jika tidak ada frame yang berhasil dibaca
    if len(frames) == 0:
        return None

    # =====================================
    # PADDING FRAME
    # =====================================
    # Jika frame kurang dari 20,
    # gunakan frame terakhir sampai jumlahnya 20

    while len(frames) < max_frames:

        frames.append(frames[-1])

    # Konversi menjadi NumPy array
    return np.array(
        frames,
        dtype=np.float32
    )


# =====================================
# LOAD CSV
# =====================================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


# =====================================
# LABEL ENCODING
# =====================================

classes = sorted(
    train_df["tag"].unique()
)

label_dict = {
    name: idx
    for idx, name in enumerate(classes)
}

print("=" * 50)
print("LABEL ENCODING")
print("=" * 50)

print(label_dict)

print("=" * 50)


# =====================================
# TRAIN DATASET
# =====================================

X_train = []
y_train = []

train_failed = 0

print()
print("Memproses TRAIN...")
print()

for index, row in train_df.iterrows():

    video_path = os.path.join(
        TRAIN_FOLDER,
        row["video_name"]
    )

    frames = extract_frames(
        video_path,
        MAX_FRAMES,
        IMG_SIZE
    )

    if frames is not None:

        X_train.append(frames)

        y_train.append(
            label_dict[row["tag"]]
        )

    else:

        train_failed += 1

    # Progress
    print(
        f"{index + 1}/{len(train_df)}",
        end="\r"
    )


# Konversi menjadi NumPy array
X_train = np.array(
    X_train,
    dtype=np.float32
)

y_train = np.array(
    y_train,
    dtype=np.int32
)


# =====================================
# TEST DATASET
# =====================================

X_test = []
y_test = []

test_failed = 0

print()
print()
print("Memproses TEST...")
print()

for index, row in test_df.iterrows():

    video_path = os.path.join(
        TEST_FOLDER,
        row["video_name"]
    )

    frames = extract_frames(
        video_path,
        MAX_FRAMES,
        IMG_SIZE
    )

    if frames is not None:

        X_test.append(frames)

        y_test.append(
            label_dict[row["tag"]]
        )

    else:

        test_failed += 1

    # Progress
    print(
        f"{index + 1}/{len(test_df)}",
        end="\r"
    )


# Konversi menjadi NumPy array
X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.int32
)


# =====================================
# MEMBUAT FOLDER DATASET
# =====================================

os.makedirs(
    "dataset_numpy",
    exist_ok=True
)


# =====================================
# SIMPAN DATASET
# =====================================

np.save(
    "dataset_numpy/X_train.npy",
    X_train
)

np.save(
    "dataset_numpy/y_train.npy",
    y_train
)

np.save(
    "dataset_numpy/X_test.npy",
    X_test
)

np.save(
    "dataset_numpy/y_test.npy",
    y_test
)


# =====================================
# INFORMASI DATASET
# =====================================

print()
print()
print("=" * 50)
print("DATASET BERHASIL DISIMPAN!")
print("=" * 50)

print()
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print()

print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

print()
print("Video TRAIN gagal :", train_failed)
print("Video TEST gagal  :", test_failed)

print()

# Ukuran dataset
train_size_mb = (
    X_train.nbytes / 1024 / 1024
)

test_size_mb = (
    X_test.nbytes / 1024 / 1024
)

print(
    "Ukuran X_train : %.2f MB"
    % train_size_mb
)

print(
    "Ukuran X_test  : %.2f MB"
    % test_size_mb
)

print()

print("Lokasi penyimpanan:")

print(
    "dataset_numpy/X_train.npy"
)

print(
    "dataset_numpy/y_train.npy"
)

print(
    "dataset_numpy/X_test.npy"
)

print(
    "dataset_numpy/y_test.npy"
)

print()

print("=" * 50)
print("TAHAP 4 SELESAI")
print("=" * 50)