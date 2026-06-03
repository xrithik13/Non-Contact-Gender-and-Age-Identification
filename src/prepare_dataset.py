"""
prepare_dataset.py  (v2 — Gender + Age)
========================================
Loads UTKFace, extracts BOTH gender and age labels,
preprocesses with MobileNetV2 normalization, and splits.

UTKFace filename format:
    [age]_[gender]_[race]_[timestamp].jpg
    e.g.  25_0_2_20170116174525125.jpg
          └─ age=25, gender=0 (male)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_PATH = "data/UTKFace"
IMG_SIZE     = 224          # MobileNetV2 native resolution
MAX_AGE      = 116.0        # oldest label in UTKFace — used to normalise age to [0,1]

# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────

images       = []
age_labels   = []
gender_labels = []

# ─────────────────────────────────────────────
# LOOP THROUGH DATASET
# ─────────────────────────────────────────────

print("Loading UTKFace dataset...")

for filename in tqdm(os.listdir(DATASET_PATH)):
    try:
        parts = filename.split('_')

        # Need at least age + gender fields
        if len(parts) < 2:
            continue

        age    = int(parts[0])
        gender = int(parts[1])

        # Sanity checks
        if gender not in [0, 1]:
            continue
        if not (1 <= age <= 116):
            continue

        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # BGR → RGB  (OpenCV reads BGR; MobileNetV2 expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 224×224
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # MobileNetV2 normalization → scales pixels to [-1, 1]
        # Do NOT use /255.0 — that's wrong for MobileNetV2
        image = preprocess_input(image.astype(np.float32))

        images.append(image)
        gender_labels.append(gender)

        # Normalise age to [0, 1] for stable regression training
        age_labels.append(age / MAX_AGE)

    except Exception:
        continue

# ─────────────────────────────────────────────
# CONVERT TO NUMPY
# ─────────────────────────────────────────────

X = np.array(images,        dtype=np.float32)
y_gender = np.array(gender_labels, dtype=np.float32)
y_age    = np.array(age_labels,    dtype=np.float32)

print(f"\nDataset loaded:")
print(f"  Images  : {X.shape}")
print(f"  Gender  : {y_gender.shape}  | Male={int((y_gender==0).sum())}  Female={int((y_gender==1).sum())}")
print(f"  Age     : min={y_age.min()*MAX_AGE:.0f}  max={y_age.max()*MAX_AGE:.0f}  mean={y_age.mean()*MAX_AGE:.1f}")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# stratify on gender so both splits are balanced
# ─────────────────────────────────────────────

X_train, X_test, \
yg_train, yg_test, \
ya_train, ya_test = train_test_split(
    X, y_gender, y_age,
    test_size=0.2,
    random_state=42,
    stratify=y_gender      # keep male/female ratio equal in both splits
)

print(f"\nTrain : {X_train.shape[0]} samples")
print(f"Test  : {X_test.shape[0]}  samples")
