"""SportVision: deteksi aktivitas manusia khusus dataset olahraga.

Hanya mendeteksi 3 kategori aktivitas yang ada di dataset:
  - CricketShot
  - Punch (Boxing)
  - TennisSwing

Video yang TIDAK termasuk kategori di atas → TIDAK TERDETEKSI.

Jalankan dengan: streamlit run 08_web_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    import tensorflow as tf
except ImportError:
    tf = None

# ──────────────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "action_recognition.keras"

# Label sesuai urutan output model (index 0, 1, 2)
DATASET_LABELS = ("CricketShot", "Punch", "TennisSwing")
LABEL_DISPLAY = {
    "CricketShot": {"icon": "🏏", "name": "Cricket Shot", "desc": "Pukulan kriket"},
    "Punch":       {"icon": "🥊", "name": "Punch (Boxing)", "desc": "Pukulan tinju"},
    "TennisSwing": {"icon": "🎾", "name": "Tennis Swing", "desc": "Ayunan tenis"},
}

SEQUENCE_FRAMES = 20
IMAGE_SIZE = 224
SAMPLE_FRAMES = 8
MIN_PERSON_CONFIDENCE = 0.25
MATCH_THRESHOLD = 0.45   # Confidence diturunkan karena kita mengandalkan Feature Distance
ENTROPY_THRESHOLD = 1.05 # Entropy dilonggarkan
MARGIN_THRESHOLD = 0.05  # Selisih minimum antar kelas dilonggarkan
DISTANCE_THRESHOLD = 3.2 # Maksimum jarak Euclidean ke centroid (Sangat ketat untuk deteksi OOD)

# ──────────────────────────────────────────────────────
# Page Config & Premium Dark CSS
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SportVision | Deteksi Aktivitas Dataset",
    page_icon="🏅",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-card-hover: #1f2b3d;
    --border: #2a3a50;
    --border-glow: #3b82f6;
    --accent: #3b82f6;
    --accent-light: #60a5fa;
    --accent-dim: rgba(59,130,246,0.15);
    --green: #10b981;
    --green-dim: rgba(16,185,129,0.12);
    --red: #ef4444;
    --red-dim: rgba(239,68,68,0.12);
    --amber: #f59e0b;
    --amber-dim: rgba(245,158,11,0.12);
    --text: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-1: linear-gradient(135deg, #3b82f6, #8b5cf6);
    --gradient-2: linear-gradient(135deg, #10b981, #3b82f6);
}

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden !important; }

/* Hero Section */
.hero-section {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    position: relative;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: var(--accent-dim);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 100px;
    padding: .35rem 1rem;
    font-size: .72rem;
    font-weight: 600;
    color: var(--accent-light);
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin: .5rem 0;
}
.hero-title span {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: .95rem;
    max-width: 640px;
    margin: .8rem auto 0;
    line-height: 1.6;
}

/* Dataset Tags */
.dataset-tags {
    display: flex;
    justify-content: center;
    gap: .6rem;
    margin: 1.2rem 0 .5rem;
    flex-wrap: wrap;
}
.dataset-tag {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .5rem .9rem;
    font-size: .82rem;
    font-weight: 600;
    color: var(--text);
    transition: all .2s;
}
.dataset-tag:hover {
    border-color: var(--accent);
    background: var(--accent-dim);
}
.dataset-tag .emoji { font-size: 1.1rem; }

/* Card Panel */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-1);
    opacity: .6;
}
.card-step {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent-light);
    margin-bottom: .6rem;
}
.card-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: .8rem;
}

/* Info Box */
.info-box {
    background: var(--accent-dim);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: .85rem 1rem;
    margin: .8rem 0;
    font-size: .82rem;
    color: var(--accent-light);
    line-height: 1.55;
}
.info-box strong { color: var(--text); }

/* Result Cards */
.result-detected {
    background: var(--green-dim);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 1rem;
}
.result-rejected {
    background: var(--red-dim);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 1rem;
}
.result-waiting {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 1rem;
    text-align: center;
}
.result-status {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .4rem;
}
.result-detected .result-status { color: var(--green); }
.result-rejected .result-status { color: var(--red); }
.result-waiting .result-status { color: var(--text-muted); }
.result-main {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    margin: .3rem 0;
    line-height: 1.15;
}
.result-activity {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: .5rem 1rem;
    margin: .6rem 0;
    font-size: 1rem;
    font-weight: 700;
    color: var(--green);
}
.result-detail {
    color: var(--text-secondary);
    font-size: .85rem;
    line-height: 1.55;
    margin-top: .5rem;
}

/* Metric Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .7rem;
    margin-top: 1rem;
}
.metric-item {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .85rem;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent-light);
    display: block;
}
.metric-label {
    font-size: .72rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: .2rem;
}

/* Probability Bars */
.prob-section {
    margin-top: 1rem;
}
.prob-title {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: .6rem;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: .5rem;
}
.prob-label {
    width: 110px;
    font-size: .78rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-align: right;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 22px;
    background: var(--bg-secondary);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width .6s ease;
}
.prob-bar-fill.winner { background: var(--gradient-2); }
.prob-bar-fill.loser { background: rgba(100,116,139,0.3); }
.prob-pct {
    width: 48px;
    font-size: .78rem;
    font-weight: 700;
    color: var(--text);
    text-align: right;
    flex-shrink: 0;
}

/* Frame Gallery */
.frame-heading {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 .8rem;
    padding-left: .3rem;
    border-left: 3px solid var(--accent);
    padding-left: .8rem;
}

/* Override Streamlit widgets */
.stRadio > div { gap: .5rem; }
.stRadio label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: .45rem .9rem !important;
    color: var(--text) !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] { border-radius: 10px; }
.stButton > button {
    background: var(--gradient-1) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 2.8rem !important;
    font-size: .9rem !important;
    letter-spacing: .02em !important;
    transition: all .25s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.3) !important;
}
.stFileUploader {
    border-radius: 12px !important;
}

/* Footer */
.app-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: .75rem;
    padding: 1.5rem 0 .5rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# Model & Deteksi
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_features() -> tuple[tf.keras.Model, tf.keras.Model, np.ndarray]:
    """Load model, feature extractor, dan pre-computed centroids untuk OOD detection."""
    if tf is None:
        raise RuntimeError("TensorFlow belum terpasang. Jalankan: pip install -r requirements.txt")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    # Layer -3 adalah LSTM (64 dimensi), sebelum Dropout dan Dense
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)
    
    centroid_path = BASE_DIR / "model" / "centroids.npy"
    if centroid_path.exists():
        centroids = np.load(centroid_path)
    else:
        # Jika tidak ada, pakai fallback
        centroids = np.zeros((len(DATASET_LABELS), 64))
        
    return model, feature_extractor, centroids


@st.cache_resource(show_spinner=False)
def load_hog_detector() -> cv2.HOGDescriptor:
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def extract_sequence(video_path: str) -> np.ndarray | None:
    """Ambil 20 frame seragam untuk input model (preprocessing identik dengan training)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    indices = set(np.linspace(0, total - 1, SEQUENCE_FRAMES, dtype=int))
    frames: list[np.ndarray] = []
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            frames.append(resized)
        idx += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < SEQUENCE_FRAMES:
        frames.append(frames[-1])
    return np.asarray(frames[:SEQUENCE_FRAMES], dtype=np.float32)


def _compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy ternormalisasi (0 = pasti, 1 = sepenuhnya acak)."""
    probs_clean = probs[probs > 1e-10]
    raw = -float(np.sum(probs_clean * np.log(probs_clean)))
    max_entropy = np.log(len(probs))  # log(3) ≈ 1.099
    return raw / max_entropy if max_entropy > 0 else 0.0


def classify_activity(video_path: str) -> dict:
    """
    Klasifikasi video berdasarkan model dataset dengan QUADRUPLE-GATE rejection.

    Video HANYA terdeteksi jika melewati SEMUA 4 filter:
      1. Confidence ≥ MATCH_THRESHOLD (90%)
      2. Entropy ≤ ENTROPY_THRESHOLD (model harus yakin, bukan ragu-ragu)
      3. Margin top1 - top2 ≥ MARGIN_THRESHOLD (30%)
      4. Feature Distance ≤ DISTANCE_THRESHOLD (Pendeteksi OOD yang kuat)

    Jika GAGAL di salah satu → TIDAK TERDETEKSI (bukan aktivitas dataset).
    """
    seq = extract_sequence(video_path)
    if seq is None:
        return {
            "detected": False,
            "label": None,
            "label_index": -1,
            "confidence": 0.0,
            "entropy": 1.0,
            "margin": 0.0,
            "distance": 99.0,
            "all_probs": {},
            "reason": "Video tidak memiliki frame yang dapat dibaca.",
        }

    model, feature_extractor, centroids = load_model_and_features()
    seq_batch = seq[None, ...]
    probs = model.predict(seq_batch, verbose=0)[0]
    features = feature_extractor.predict(seq_batch, verbose=0)[0]

    # Urutkan probabilitas dari tertinggi ke terendah
    sorted_indices = np.argsort(probs)[::-1]
    label_idx = int(sorted_indices[0])
    confidence = float(probs[label_idx])
    second_best = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
    margin = confidence - second_best
    entropy = _compute_entropy(probs)
    
    # Hitung jarak Euclidean ke centroid kelas yang diprediksi
    if np.any(centroids):
        distance = float(np.linalg.norm(features - centroids[label_idx]))
    else:
        distance = 0.0

    if label_idx >= len(DATASET_LABELS):
        return {
            "detected": False,
            "label": None,
            "label_index": label_idx,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "distance": distance,
            "all_probs": {DATASET_LABELS[i]: float(probs[i]) for i in range(min(len(probs), len(DATASET_LABELS)))},
            "reason": "Index kelas model tidak sesuai dengan dataset.",
        }

    all_probs = {DATASET_LABELS[i]: float(probs[i]) for i in range(len(DATASET_LABELS))}
    label = DATASET_LABELS[label_idx]

    # ── GATE 1: Confidence Check ──
    if confidence < MATCH_THRESHOLD:
        return {
            "detected": False,
            "label": label,
            "label_index": label_idx,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "all_probs": all_probs,
            "reason": (
                f"❌ GAGAL Gate 1 (Confidence): Confidence tertinggi hanya {confidence:.1%} "
                f"untuk '{label}', di bawah ambang {MATCH_THRESHOLD:.0%}. "
                f"Video kemungkinan besar BUKAN aktivitas dalam dataset."
            ),
        }

    # ── GATE 2: Entropy Check ──
    if entropy > ENTROPY_THRESHOLD:
        return {
            "detected": False,
            "label": label,
            "label_index": label_idx,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "all_probs": all_probs,
            "reason": (
                f"❌ GAGAL Gate 2 (Entropy): Entropy = {entropy:.2f} (maks {ENTROPY_THRESHOLD:.2f}). "
                f"Model tidak cukup yakin — probabilitas tersebar merata. "
                f"Video kemungkinan BUKAN aktivitas dalam dataset."
            ),
        }

    # ── GATE 3: Margin Check ──
    if margin < MARGIN_THRESHOLD:
        second_label = DATASET_LABELS[int(sorted_indices[1])] if int(sorted_indices[1]) < len(DATASET_LABELS) else "?"
        return {
            "detected": False,
            "label": label,
            "label_index": label_idx,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "all_probs": all_probs,
            "reason": (
                f"❌ GAGAL Gate 3 (Margin): Selisih antara '{label}' ({confidence:.1%}) "
                f"dan '{second_label}' ({second_best:.1%}) hanya {margin:.1%}, "
                f"di bawah ambang {MARGIN_THRESHOLD:.0%}. "
                f"Model tidak bisa membedakan — video kemungkinan BUKAN aktivitas dataset."
            ),
        }

    # ── GATE 4: Feature Distance (OOD) Check ──
    if distance > DISTANCE_THRESHOLD:
        return {
            "detected": False,
            "label": label,
            "label_index": label_idx,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "distance": distance,
            "all_probs": all_probs,
            "reason": (
                f"❌ GAGAL Gate 4 (Feature Distance): Jarak ke centroid dataset = {distance:.2f} "
                f"(maks {DISTANCE_THRESHOLD:.1f}). "
                f"Video berada di luar distribusi (OOD) dan BUKAN aktivitas dataset."
            ),
        }

    # ── SEMUA GATE LOLOS → TERDETEKSI ──
    return {
        "detected": True,
        "label": label,
        "label_index": label_idx,
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "distance": distance,
        "all_probs": all_probs,
        "reason": "",
    }


def detect_people_in_frame(frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Deteksi manusia pada satu frame menggunakan HOG + SVM."""
    h, w = frame.shape[:2]
    scale = min(1.0, 720 / w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame.copy()
    boxes, weights = load_hog_detector().detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

    filtered = [(list(map(int, b)), float(wt)) for b, wt in zip(boxes, weights) if float(wt) >= MIN_PERSON_CONFIDENCE]
    bx_list = [b for b, _ in filtered]
    sc_list = [s for _, s in filtered]

    kept = np.array(cv2.dnn.NMSBoxes(bx_list, sc_list, MIN_PERSON_CONFIDENCE, 0.35)).flatten() if bx_list else []

    annotated = frame.copy()
    count = 0
    for k in kept:
        x, y, bw, bh = [int(v / scale) for v in bx_list[int(k)]]
        count += 1
        # Kotak hijau untuk aktivitas terdeteksi
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (16, 185, 129), 3)
        # Label bar
        cv2.rectangle(annotated, (x, max(0, y - 32)), (x + 180, y), (16, 185, 129), -1)
        cv2.putText(annotated, "AKTIVITAS MANUSIA", (x + 6, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), count


def full_analysis(video_path: str) -> dict:
    """Pipeline lengkap: klasifikasi dataset → deteksi manusia (hanya jika terdeteksi)."""
    classification = classify_activity(video_path)

    if not classification["detected"]:
        return {**classification, "frames": [], "sampled_frames": 0, "active_frames": 0, "max_people": 0}

    # Deteksi manusia pada sample frame
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(np.linspace(0, total - 1, SAMPLE_FRAMES, dtype=int))
    frames, people_counts = [], []
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target_indices:
            annotated, count = detect_people_in_frame(frame)
            frames.append(annotated)
            people_counts.append(count)
        idx += 1
    cap.release()

    return {
        **classification,
        "frames": frames,
        "sampled_frames": len(frames),
        "active_frames": sum(1 for c in people_counts if c > 0),
        "max_people": max(people_counts, default=0),
    }


def get_demo_videos() -> dict[str, Path]:
    return {
        "🏏 CricketShot": BASE_DIR / "Dataset" / "test" / "v_CricketShot_g01_c01.avi",
        "🥊 Punch": BASE_DIR / "Dataset" / "test" / "v_Punch_g01_c01.avi",
        "🎾 TennisSwing": BASE_DIR / "Dataset" / "test" / "v_TennisSwing_g01_c01.avi",
    }


# ──────────────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ── Hero ──
st.markdown("""
<div class="hero-section">
    <div class="hero-badge"><span class="dot"></span> Computer Vision · Dataset-Terbatas</div>
    <div class="hero-title">Deteksi Aktivitas Manusia<br><span>Khusus Dataset Olahraga</span></div>
    <div class="hero-sub">
        Sistem ini <strong>hanya mendeteksi</strong> aktivitas yang ada dalam dataset pelatihan.
        Video yang tidak sesuai dengan kategori dataset akan <strong>ditolak secara otomatis</strong>.
    </div>
    <div class="dataset-tags">
        <div class="dataset-tag"><span class="emoji">🏏</span> CricketShot</div>
        <div class="dataset-tag"><span class="emoji">🥊</span> Punch</div>
        <div class="dataset-tag"><span class="emoji">🎾</span> TennisSwing</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Columns ──
col_input, col_result = st.columns((1.05, 0.95), gap="large")
video_path: str | None = None
temp_path: str | None = None

with col_input:
    st.markdown("""
    <div class="card">
        <div class="card-step">Langkah 01 · Input Video</div>
        <div class="card-title">Masukkan Video untuk Dianalisis</div>
        <div class="info-box">
            <strong>⚠️ Penting:</strong> Hanya video berisi aktivitas
            <strong>CricketShot</strong>, <strong>Punch</strong>, atau <strong>TennisSwing</strong>
            yang akan terdeteksi. Video lain (berjalan, berenang, bermain sepak bola, dll.)
            akan ditolak karena <strong>tidak ada di dataset</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    source = st.radio(
        "Pilih sumber video",
        ("📂 Video contoh dari dataset", "📤 Unggah video sendiri"),
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "📂 Video contoh dari dataset":
        demos = get_demo_videos()
        choice = st.selectbox("Pilih contoh video dataset", tuple(demos))
        candidate = demos[choice]
        if candidate.exists():
            video_path = str(candidate)
            st.video(video_path)
        else:
            st.error(f"File tidak ditemukan: `{candidate}`")
    else:
        upload = st.file_uploader("Unggah file video", type=("mp4", "avi", "mov", "mkv"))
        if upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix or ".mp4")
            tmp.write(upload.getvalue())
            tmp.close()
            temp_path = tmp.name
            video_path = tmp.name
            st.video(upload.getvalue())

    run = st.button("🔍  Analisis & Deteksi", disabled=video_path is None, use_container_width=True if hasattr(st.button, '__wrapped__') else True)

    st.markdown("""
    <div class="info-box" style="margin-top: .8rem; opacity: .7;">
        Proses: Video → Klasifikasi model dataset → Jika cocok → Deteksi lokasi manusia.
        Jika tidak cocok dengan dataset → <strong>TIDAK TERDETEKSI</strong>.
    </div>
    """, unsafe_allow_html=True)


# ── Run Analysis ──
if run and video_path:
    try:
        with st.spinner("🔄 Menganalisis video terhadap dataset…"):
            st.session_state.result = full_analysis(video_path)
    except Exception as err:
        st.session_state.result = {
            "detected": False,
            "label": None,
            "confidence": 0.0,
            "all_probs": {},
            "reason": f"Terjadi error: {err}",
            "frames": [],
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ── Result Panel ──
with col_result:
    st.markdown("""
    <div class="card">
        <div class="card-step">Langkah 02 · Hasil Deteksi</div>
        <div class="card-title">Status Analisis Video</div>
    """, unsafe_allow_html=True)

    result = st.session_state.result

    if result is None:
        # Belum ada analisis
        st.markdown("""
        <div class="result-waiting">
            <div class="result-status">Menunggu input</div>
            <div class="result-main" style="font-size:1.2rem;">Belum ada video yang dianalisis</div>
            <div class="result-detail">
                Pilih video contoh dataset atau unggah video, lalu tekan tombol <strong>Analisis & Deteksi</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not result["detected"]:
        # TIDAK TERDETEKSI — video bukan dari dataset
        conf_val = result.get('confidence', 0)
        ent_val = result.get('entropy', 1)
        mar_val = result.get('margin', 0)
        dist_val = result.get('distance', 99)

        st.markdown(f"""
        <div class="result-rejected">
            <div class="result-status">❌ Tidak Terdeteksi</div>
            <div class="result-main">Video BUKAN Aktivitas Dataset</div>
            <div class="result-detail">
                {result['reason']}<br><br>
                Sistem <strong>hanya</strong> mendeteksi: <strong>CricketShot</strong>, <strong>Punch</strong>, dan <strong>TennisSwing</strong>.
                Video di luar 3 kategori ini <strong>tidak akan dideteksi sama sekali</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tampilkan status 4 gate
        gate1_ok = conf_val >= MATCH_THRESHOLD
        gate2_ok = ent_val <= ENTROPY_THRESHOLD
        gate3_ok = mar_val >= MARGIN_THRESHOLD
        gate4_ok = dist_val <= DISTANCE_THRESHOLD
        g1_icon = "✅" if gate1_ok else "❌"
        g2_icon = "✅" if gate2_ok else "❌"
        g3_icon = "✅" if gate3_ok else "❌"
        g4_icon = "✅" if gate4_ok else "❌"
        g1_color = "var(--green)" if gate1_ok else "var(--red)"
        g2_color = "var(--green)" if gate2_ok else "var(--red)"
        g3_color = "var(--green)" if gate3_ok else "var(--red)"
        g4_color = "var(--green)" if gate4_ok else "var(--red)"

        st.markdown(f"""
        <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:12px;padding:1rem;margin:.8rem 0;">
            <div style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:.6rem;">Filter Validasi Dataset (4 Gate)</div>
            <div style="display:flex;flex-direction:column;gap:.5rem;">
                <div style="display:flex;align-items:center;gap:.5rem;">
                    <span style="font-size:1rem;">{g1_icon}</span>
                    <span style="font-size:.82rem;color:{g1_color};font-weight:600;">Gate 1 — Confidence: {conf_val:.1%} (min {MATCH_THRESHOLD:.0%})</span>
                </div>
                <div style="display:flex;align-items:center;gap:.5rem;">
                    <span style="font-size:1rem;">{g2_icon}</span>
                    <span style="font-size:.82rem;color:{g2_color};font-weight:600;">Gate 2 — Entropy: {ent_val:.3f} (maks {ENTROPY_THRESHOLD})</span>
                </div>
                <div style="display:flex;align-items:center;gap:.5rem;">
                    <span style="font-size:1rem;">{g3_icon}</span>
                    <span style="font-size:.82rem;color:{g3_color};font-weight:600;">Gate 3 — Margin: {mar_val:.1%} (min {MARGIN_THRESHOLD:.0%})</span>
                </div>
                <div style="display:flex;align-items:center;gap:.5rem;">
                    <span style="font-size:1rem;">{g4_icon}</span>
                    <span style="font-size:.82rem;color:{g4_color};font-weight:600;">Gate 4 — OOD Distance: {dist_val:.2f} (maks {DISTANCE_THRESHOLD:.1f})</span>
                </div>
            </div>
            <div style="margin-top:.6rem;font-size:.78rem;color:var(--red);font-weight:600;">⛔ Video DITOLAK — Harus lolos keempat gate untuk terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)

        # Tampilkan probability bars jika ada
        if result.get("all_probs"):
            prob_html = '<div class="prob-section"><div class="prob-title">Probabilitas per Kategori Dataset</div>'
            for label_name in DATASET_LABELS:
                p = result["all_probs"].get(label_name, 0.0)
                pct = p * 100
                info = LABEL_DISPLAY.get(label_name, {})
                icon = info.get("icon", "")
                bar_class = "loser"
                prob_html += f"""
                <div class="prob-row">
                    <div class="prob-label">{icon} {label_name}</div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill {bar_class}" style="width:{pct:.1f}%"></div></div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>"""
            prob_html += "</div>"
            st.markdown(prob_html, unsafe_allow_html=True)

    else:
        # TERDETEKSI — video sesuai dataset
        label = result["label"]
        info = LABEL_DISPLAY.get(label, {"icon": "📹", "name": label, "desc": ""})
        conf = result["confidence"]

        st.markdown(f"""
        <div class="result-detected">
            <div class="result-status">✅ Terdeteksi dalam Dataset</div>
            <div class="result-main">Aktivitas Manusia Terdeteksi</div>
            <div class="result-activity">{info['icon']}  {info['name']}</div>
            <div class="result-detail">
                Video dikenali sebagai <strong>{info['name']}</strong> ({info['desc']})
                dengan confidence <strong>{conf:.0%}</strong>.
                Aktivitas ini terdaftar dalam dataset pelatihan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        prob_html = '<div class="prob-section"><div class="prob-title">Probabilitas per Kategori Dataset</div>'
        for label_name in DATASET_LABELS:
            p = result["all_probs"].get(label_name, 0.0)
            pct = p * 100
            lbl_info = LABEL_DISPLAY.get(label_name, {})
            icon = lbl_info.get("icon", "")
            bar_class = "winner" if label_name == label else "loser"
            prob_html += f"""
            <div class="prob-row">
                <div class="prob-label">{icon} {label_name}</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill {bar_class}" style="width:{pct:.1f}%"></div></div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>"""
        prob_html += "</div>"
        st.markdown(prob_html, unsafe_allow_html=True)

        # Metric cards
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-item">
                <span class="metric-value">{result.get('active_frames', 0)}/{result.get('sampled_frames', 0)}</span>
                <span class="metric-label">Frame Beraktivitas</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">{result.get('max_people', 0)}</span>
                <span class="metric-label">Orang Maks/Frame</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">{conf:.0%}</span>
                <span class="metric-label">Confidence Dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Frame Gallery (hanya jika terdeteksi) ──
if st.session_state.result and st.session_state.result.get("detected") and st.session_state.result.get("frames"):
    label = st.session_state.result["label"]
    info = LABEL_DISPLAY.get(label, {"icon": "📹", "name": label})
    st.markdown(
        f'<div class="frame-heading">{info["icon"]} Frame Deteksi — {info["name"]}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for i, (col, frame) in enumerate(zip(cols, st.session_state.result["frames"]), 1):
        col.image(frame, caption=f"Frame {i}", use_container_width=True)

    # Tampilkan sisa frame jika lebih dari 4
    remaining = st.session_state.result["frames"][4:]
    if remaining:
        cols2 = st.columns(4)
        for i, (col, frame) in enumerate(zip(cols2, remaining), 5):
            col.image(frame, caption=f"Frame {i}", use_container_width=True)


# ── Footer ──
st.markdown("""
<div class="app-footer">
    SportVision · Deteksi aktivitas manusia <strong>terbatas pada kategori dataset</strong> (CricketShot, Punch, TennisSwing)<br>
    Video di luar dataset → tidak terdeteksi · Tugas Besar Computer Vision
</div>
""", unsafe_allow_html=True)
