"""
evaluate_model.py
==================
Loads the saved gender_age_model.h5 and properly evaluates it.
Run this once to get the real accuracy and MAE numbers.

    python src/evaluate_model.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_PATH    = "data/UTKFace"
MODEL_SAVE_PATH = "models/gender_age_model.h5"
IMG_SIZE        = 224
MAX_AGE         = 116.0

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("Loading model...")
model = load_model(MODEL_SAVE_PATH, compile=False)
model.compile(
    optimizer="adam",
    loss={"gender_output": "binary_crossentropy", "age_output": "mae"},
    metrics={"gender_output": "accuracy", "age_output": "mae"}
)
print("Done.\n")

# ─────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────

print("Loading UTKFace dataset...")

images, gender_labels, age_labels = [], [], []

for filename in tqdm(os.listdir(DATASET_PATH), desc="  Reading", unit="img"):
    try:
        parts  = filename.split('_')
        if len(parts) < 2:
            continue
        age    = int(parts[0])
        gender = int(parts[1])
        if gender not in [0, 1]:
            continue
        if not (1 <= age <= 116):
            continue
        image = cv2.imread(os.path.join(DATASET_PATH, filename))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = preprocess_input(image.astype(np.float32))
        images.append(image)
        gender_labels.append(float(gender))
        age_labels.append(age / MAX_AGE)
    except Exception:
        continue

X        = np.array(images,        dtype=np.float32)
y_gender = np.array(gender_labels, dtype=np.float32)
y_age    = np.array(age_labels,    dtype=np.float32)

# Same split as training — same random_state=42 gives same test set
_, X_test, _, yg_test, _, ya_test = train_test_split(
    X, y_gender, y_age,
    test_size=0.2,
    random_state=42,
    stratify=y_gender
)

print(f"  Test samples: {X_test.shape[0]}\n")

# ─────────────────────────────────────────────
# PREDICT ON FULL TEST SET
# ─────────────────────────────────────────────

print("Running predictions on test set...")
preds = model.predict(X_test, batch_size=64, verbose=1)

gender_preds = preds[0].flatten()     # shape (N,)
age_preds    = preds[1].flatten() * MAX_AGE   # de-normalise to years
age_preds    = np.clip(age_preds, 1, MAX_AGE)

# ─────────────────────────────────────────────
# COMPUTE METRICS MANUALLY — no index guessing
# ─────────────────────────────────────────────

# Gender accuracy
gender_pred_labels = (gender_preds > 0.5).astype(float)
gender_accuracy    = (gender_pred_labels == yg_test).mean()

# Age MAE in years
age_true_years = ya_test * MAX_AGE
age_mae        = np.abs(age_preds - age_true_years).mean()

# Gender per-class accuracy
male_mask   = yg_test == 0
female_mask = yg_test == 1
male_acc    = (gender_pred_labels[male_mask]   == 0).mean()
female_acc  = (gender_pred_labels[female_mask] == 1).mean()

# Age MAE per gender
male_age_mae   = np.abs(age_preds[male_mask]   - age_true_years[male_mask]).mean()
female_age_mae = np.abs(age_preds[female_mask] - age_true_years[female_mask]).mean()

# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────

print(f"\n  ┌──────────────────────────────────────────┐")
print(f"  │           EVALUATION RESULTS             │")
print(f"  ├──────────────────────────────────────────┤")
print(f"  │  Overall Gender Accuracy : {gender_accuracy*100:6.2f}%        │")
print(f"  │    Male   Accuracy       : {male_acc*100:6.2f}%        │")
print(f"  │    Female Accuracy       : {female_acc*100:6.2f}%        │")
print(f"  ├──────────────────────────────────────────┤")
print(f"  │  Overall Age MAE         : {age_mae:6.2f} years     │")
print(f"  │    Male   Age MAE        : {male_age_mae:6.2f} years     │")
print(f"  │    Female Age MAE        : {female_age_mae:6.2f} years     │")
print(f"  └──────────────────────────────────────────┘")

# ─────────────────────────────────────────────
# SAMPLE PREDICTIONS — 10 random test images
# ─────────────────────────────────────────────

print(f"\n  Sample predictions (10 random test images):")
print(f"  {'#':<4} {'True Gender':<14} {'Pred Gender':<14} {'Conf':<8} {'True Age':<10} {'Pred Age':<10} {'Match'}")
print(f"  {'-'*72}")

rng     = np.random.default_rng(0)
indices = rng.choice(len(X_test), size=10, replace=False)

for i, idx in enumerate(indices):
    g_pred   = float(gender_preds[idx])
    a_pred   = float(age_preds[idx])
    g_true   = int(yg_test[idx])
    a_true   = ya_test[idx] * MAX_AGE
    g_label  = "Female" if g_pred > 0.5 else "Male"
    g_true_l = "Female" if g_true == 1  else "Male"
    conf     = g_pred if g_pred > 0.5 else 1 - g_pred
    match    = "✓" if g_label == g_true_l else "✗"
    print(f"  {i:<4} {g_true_l:<14} {g_label:<14} {conf*100:5.1f}%   "
          f"{a_true:5.0f} yrs   {a_pred:5.1f} yrs   {match}")

print("\nDone.")
