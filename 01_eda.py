from pathlib import Path
import re

import pandas as pd


def load_train_metadata(csv_path="train.csv", video_dir="Dataset/train"):
    """Load train.csv, or recover its metadata from the video filenames.

    The expected filename format is ``v_<label>_g<number>_c<number>.avi``.
    """
    try:
        dataframe = pd.read_csv(csv_path)
        if set(("video_name", "tag")) <= set(dataframe.columns):
            return dataframe
        raise ValueError("Kolom 'video_name' dan 'tag' tidak ditemukan.")
    except (pd.errors.ParserError, ValueError) as error:
        pattern = re.compile(r"^v_(?P<tag>.+?)_g\d+_c\d+\.avi$")
        records = []

        for video_path in sorted(Path(video_dir).glob("*.avi")):
            match = pattern.match(video_path.name)
            if match:
                records.append(
                    {"video_name": video_path.name, "tag": match.group("tag")}
                )

        if not records:
            raise RuntimeError(
                f"Tidak dapat membaca {csv_path} dan tidak ada metadata video di {video_dir}."
            ) from error

        print(
            f"PERINGATAN: {csv_path} rusak ({error}). "
            f"Menggunakan {len(records)} metadata yang dibuat dari {video_dir}."
        )
        return pd.DataFrame(records)


train_df = load_train_metadata()
test_df = pd.read_csv("test.csv")

print("=" * 50)
print("INFORMASI DATASET")
print("=" * 50)

print(f"Jumlah data train : {len(train_df)}")
print(f"Jumlah data test  : {len(test_df)}")

print()

print("=" * 50)
print("3 DATA PERTAMA TRAIN")
print("=" * 50)

print(train_df.head())

print()

print("=" * 50)
print("JUMLAH KELAS")
print("=" * 50)

print(train_df["tag"].value_counts())

print()

print("=" * 50)
print("JUMLAH KELAS UNIK")
print("=" * 50)

print(train_df["tag"].nunique())

print()

print("=" * 50)
print("DAFTAR KELAS")
print("=" * 50)

print(sorted(train_df["tag"].unique()))
