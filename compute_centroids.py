import numpy as np
import tensorflow as tf
from pathlib import Path
import os

model = tf.keras.models.load_model('model/action_recognition.keras')
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

X_train = np.load('dataset_numpy/X_train.npy', mmap_mode='r')
y_train = np.load('dataset_numpy/y_train.npy')

print("Extracting features for training set...")
# Sample some training data to compute centroids
sampled_indices = []
for c in range(3):
    idx_c = np.where(y_train == c)[0]
    # take up to 50 samples per class to be fast
    sampled_indices.extend(idx_c[:50])

X_sample = X_train[sampled_indices]
features = feature_extractor.predict(X_sample, batch_size=4)

centroids = []
for c in range(3):
    c_features = features[y_train[sampled_indices] == c]
    centroids.append(np.mean(c_features, axis=0))
centroids = np.array(centroids)
np.save('model/centroids.npy', centroids)

# Compute max distance for training samples to their centroid
max_distances = []
for c in range(3):
    c_features = features[y_train[sampled_indices] == c]
    dists = np.linalg.norm(c_features - centroids[c], axis=1)
    max_distances.append(np.max(dists))
    print(f"Class {c} max distance: {np.max(dists)}")

print("Max distance threshold to use:", np.max(max_distances))
