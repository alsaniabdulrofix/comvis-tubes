import cv2
import pandas as pd
import os
import re
from pathlib import Path


# =====================================
# KONFIGURASI
# =====================================

CSV_FILE = "train.csv"
VIDEO_FOLDER = "Dataset/train"

# Pilih video yang ingin diperiksa
VIDEO_NAME = "v_TennisSwing_g11_c01.avi"


# =====================================
# MEMBACA CSV
# =====================================

try:
    train_df = pd.read_csv(CSV_FILE)
except pd.errors.ParserError as error:
    # train.csv dapat dibuat ulang dari format nama file video.
    pattern = re.compile(r"^v_(?P<tag>.+?)_g\d+_c\d+\.avi$")
    records = []

    for video_file in sorted(Path(VIDEO_FOLDER).glob("*.avi")):
        match = pattern.match(video_file.name)
        if match:
            records.append(
                {"video_name": video_file.name, "tag": match.group("tag")}
            )

    if not records:
        raise RuntimeError(
            f"{CSV_FILE} rusak dan tidak ada video .avi di {VIDEO_FOLDER}."
        ) from error

    print(
        f"PERINGATAN: {CSV_FILE} rusak; "
        f"menggunakan {len(records)} metadata dari {VIDEO_FOLDER}."
    )
    train_df = pd.DataFrame(records)


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
