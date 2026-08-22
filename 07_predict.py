"""Evaluate the saved action-recognition model on the complete test set."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report


MODEL_PATH = "model/action_recognition.keras"
DATASET_DIR = "dataset_numpy"
BATCH_SIZE = 4

# The order must match the label encoding used when dataset_numpy was created.
CLASSES = ["Cricket", "Boxing", "Tennis"]


def test_generator(features: np.ndarray):
    """Yield one test sample at a time without loading the whole array into RAM."""
    for feature in features:
        yield feature


def format_percentage(value: float) -> str:
    """Format a fraction using Indonesian decimal notation."""
    return f"{value * 100:.2f}%".replace(".", ",")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    print("=" * 58)
    print("EVALUASI MODEL PADA DATA TEST")
    print("=" * 58)

    model = tf.keras.models.load_model(MODEL_PATH)
    x_test = np.load(
        os.path.join(DATASET_DIR, "X_test.npy"), mmap_mode="r"
    )
    y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

    if model.output_shape[-1] != len(CLASSES):
        raise ValueError(
            f"Model memiliki {model.output_shape[-1]} output, "
            f"tetapi konfigurasi memiliki {len(CLASSES)} kelas."
        )

    if not set(np.unique(y_test)).issubset(set(range(len(CLASSES)))):
        raise ValueError(
            "Label y_test tidak sesuai dengan urutan CLASSES: "
            f"{np.unique(y_test)}"
        )

    dataset = tf.data.Dataset.from_generator(
        lambda: test_generator(x_test),
        output_signature=tf.TensorSpec(
            shape=(20, 224, 224, 3), dtype=tf.float32
        ),
    ).batch(BATCH_SIZE)

    probabilities = model.predict(
        dataset, steps=math.ceil(len(y_test) / BATCH_SIZE), verbose=0
    )
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for class_name in CLASSES:
        metrics = report[class_name]
        rows.append(
            {
                "Classes": class_name,
                # Accuracy per class in the requested table is precision.
                "Accuracy": format_percentage(metrics["precision"]),
                "Recall": format_percentage(metrics["recall"]),
                "F1 Score": format_percentage(metrics["f1-score"]),
            }
        )

    macro = report["macro avg"]
    rows.append(
        {
            "Classes": "Macro Average",
            "Accuracy": format_percentage(macro["precision"]),
            "Recall": format_percentage(macro["recall"]),
            "F1 Score": format_percentage(macro["f1-score"]),
        }
    )

    print("\nClasses        Accuracy    Recall      F1 Score")
    print("-" * 58)
    print(pd.DataFrame(rows).to_string(index=False, header=False))
    print("\nCatatan: 'Accuracy' per kelas pada tabel ini adalah precision; "
          "accuracy secara statistik hanya berlaku untuk seluruh model.")


if __name__ == "__main__":
    main()
