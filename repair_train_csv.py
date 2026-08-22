"""Rebuild corrupted training metadata from the filenames in Dataset/train."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


CSV_PATH = Path("train.csv")
BACKUP_PATH = Path("train.csv.corrupted-backup")
VIDEO_DIR = Path("Dataset/train")
FILENAME_PATTERN = re.compile(r"^v_(?P<tag>.+?)_g\d+_c\d+\.avi$")


def main() -> None:
    videos = []
    for video_path in sorted(VIDEO_DIR.glob("*.avi")):
        match = FILENAME_PATTERN.match(video_path.name)
        if match:
            videos.append((video_path.name, match.group("tag")))

    if not videos:
        raise RuntimeError(f"Tidak ada video yang cocok di {VIDEO_DIR}.")

    if CSV_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("video_name", "tag"))
        writer.writerows(videos)

    print(f"{CSV_PATH} berhasil dibuat ulang: {len(videos)} video.")
    print(f"Backup file rusak: {BACKUP_PATH}")


if __name__ == "__main__":
    main()
