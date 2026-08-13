import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# KONFIGURASI
# =====================================================

DATASET_PATH = "train"
CSV_FILE = "train.csv"

IMG_SIZE = 224
MAX_FRAMES = 20


# =====================================================
# MEMBACA CSV
# =====================================================

df = pd.read_csv(CSV_FILE)

print("=" * 50)
print("Jumlah Video :", len(df))
print("=" * 50)


# =====================================================
# INPUT NAMA VIDEO
# =====================================================

video_name = input(
    "\nMasukkan nama video (.avi): "
).strip()


# =====================================================
# CEK VIDEO
# =====================================================

if video_name not in df["video_name"].values:

    print("\nVideo tidak ditemukan pada train.csv")

    exit()


video_path = os.path.join(
    DATASET_PATH,
    video_name
)


# =====================================================
# FUNGSI MENGAMBIL FRAME
# =====================================================

def extract_frames(
    video_path,
    max_frames=20,
    img_size=224
):

    cap = cv2.VideoCapture(video_path)

    # Mendapatkan jumlah frame video
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Jika video tidak memiliki frame
    if frame_count == 0:

        cap.release()

        return np.array([])


    # =================================================
    # Menentukan frame yang akan diambil
    # =================================================

    frame_indices = np.linspace(
        0,
        frame_count - 1,
        max_frames,
        dtype=int
    )


    frames = []

    current = 0


    # =================================================
    # Membaca video
    # =================================================

    while cap.isOpened():

        ret, frame = cap.read()


        if not ret:

            break


        # Jika frame sekarang termasuk
        # frame yang ingin diambil

        if current in frame_indices:


            # =========================================
            # BGR -> RGB
            # =========================================

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # =========================================
            # Resize
            # =========================================

            frame = cv2.resize(
                frame,
                (img_size, img_size)
            )


            # =========================================
            # Normalisasi
            # =========================================

            frame = frame.astype(
                np.float32
            ) / 255.0


            frames.append(frame)


        current += 1


    cap.release()


    # =================================================
    # Jika tidak ada frame
    # =================================================

    if len(frames) == 0:

        return np.array([])


    # =================================================
    # PADDING
    # =================================================
    # Jika frame yang berhasil dibaca kurang dari
    # MAX_FRAMES, gunakan frame terakhir.

    while len(frames) < max_frames:

        frames.append(frames[-1])


    # =================================================
    # Konversi menjadi NumPy Array
    # =================================================

    return np.array(
        frames,
        dtype=np.float32
    )


# =====================================================
# PROSES PREPROCESSING
# =====================================================

frames = extract_frames(
    video_path,
    MAX_FRAMES,
    IMG_SIZE
)


# =====================================================
# INFORMASI HASIL
# =====================================================

print()

print("=" * 50)
print("HASIL PREPROCESSING")
print("=" * 50)

print(
    "Nama Video :",
    video_name
)


# =====================================================
# CEK FRAME
# =====================================================

if len(frames) == 0:

    print("Frame gagal diekstrak.")

    exit()


# =====================================================
# INFORMASI DATA
# =====================================================

print(
    "Shape Frame :",
    frames.shape
)

print(
    "Tipe Data :",
    frames.dtype
)

print(
    "Nilai Minimum :",
    frames.min()
)

print(
    "Nilai Maksimum :",
    frames.max()
)


# =====================================================
# CONTOH NILAI PIXEL
# =====================================================

print()

print("Contoh Nilai Pixel")

print(
    frames[10][100][100]
)


# =====================================================
# INFORMASI JUMLAH FRAME
# =====================================================

print()

print(
    "Target Frame :",
    MAX_FRAMES
)

print(
    "Jumlah Frame Berhasil Diambil :",
    len(frames)
)


# =====================================================
# CEK JUMLAH FRAME
# =====================================================

if len(frames) != MAX_FRAMES:

    print(
        "WARNING : Jumlah frame tidak sesuai!"
    )

else:

    print(
        "Status : Jumlah frame sesuai target."
    )


# =====================================================
# VISUALISASI 5 FRAME PERTAMA
# =====================================================

plt.figure(
    figsize=(16, 5)
)


for i in range(
    min(5, len(frames))
):

    plt.subplot(
        1,
        5,
        i + 1
    )

    plt.imshow(
        frames[i]
    )

    plt.title(
        f"Frame {i + 1}"
    )

    plt.axis("off")


plt.suptitle(
    "Hasil Frame Extraction",
    fontsize=16
)

plt.tight_layout()

plt.show()