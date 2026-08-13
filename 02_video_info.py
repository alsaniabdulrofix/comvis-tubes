import cv2
import pandas as pd
import os


# =====================================
# KONFIGURASI
# =====================================

CSV_FILE = "train.csv"
VIDEO_FOLDER = "train"

# Pilih video yang ingin diperiksa
VIDEO_NAME = "v_TennisSwing_g23_c01.avi"


# =====================================
# MEMBACA CSV
# =====================================

train_df = pd.read_csv(CSV_FILE)


# =====================================
# CEK VIDEO DI CSV
# =====================================

if VIDEO_NAME not in train_df["video_name"].values:

    print("Video tidak ditemukan di train.csv")
    exit()


# =====================================
# PATH VIDEO
# =====================================

video_path = os.path.join(
    VIDEO_FOLDER,
    VIDEO_NAME
)


# =====================================
# INFORMASI VIDEO
# =====================================

print("=" * 50)
print("Informasi Video")
print("=" * 50)

print("Nama Video :", VIDEO_NAME)


# =====================================
# MEMBUKA VIDEO
# =====================================

cap = cv2.VideoCapture(video_path)


if not cap.isOpened():

    print("Video gagal dibuka")

    exit()


# =====================================
# MENGAMBIL INFORMASI VIDEO
# =====================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# =====================================
# MENGHITUNG DURASI
# =====================================

if fps > 0:

    duration = frame_count / fps

else:

    duration = 0


# =====================================
# MENAMPILKAN HASIL
# =====================================

print(
    f"FPS          : {fps}"
)

print(
    f"Resolusi     : {width} x {height}"
)

print(
    f"Jumlah Frame : {frame_count}"
)

print(
    f"Durasi       : {duration:.2f} detik"
)


# =====================================
# MENUTUP VIDEO
# =====================================

cap.release()