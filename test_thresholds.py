import os
import cv2
import numpy as np
import tensorflow as tf

SEQUENCE_FRAMES = 20
IMAGE_SIZE = 224

def extract_sequence(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) == SEQUENCE_FRAMES:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized / 255.0
        frames.append(normalized)

    cap.release()
    if not frames:
        return None
    while len(frames) < SEQUENCE_FRAMES:
        frames.append(frames[-1])
    return np.asarray(frames[:SEQUENCE_FRAMES], dtype=np.float32)


model = tf.keras.models.load_model('model/action_recognition.keras')
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)
centroids = np.load('model/centroids.npy')

import glob
samples = [
    'Dataset/test\\v_CricketShot_g01_c01.avi',
    'Dataset/test\\v_Punch_g01_c01.avi',
    'Dataset/test\\v_TennisSwing_g01_c01.avi',
    'Dataset/test\\v_CricketShot_g04_c04.avi',
    'Dataset/test\\v_Punch_g04_c04.avi',
    'Dataset/test\\v_TennisSwing_g04_c04.avi'
]

for v in samples:
    if os.path.exists(v):
        seq = extract_sequence(v)
        seq_batch = seq[None, ...]
        probs = model.predict(seq_batch, verbose=0)[0]
        features = feature_extractor.predict(seq_batch, verbose=0)[0]
        sorted_indices = np.argsort(probs)[::-1]
        label_idx = int(sorted_indices[0])
        dist = float(np.linalg.norm(features - centroids[label_idx]))
        
        print(f"Video: {v}")
        print(f"  Max Prob: {float(probs[label_idx]):.4f}")
        print(f"  Distance: {dist:.4f}")
        print("-" * 40)
