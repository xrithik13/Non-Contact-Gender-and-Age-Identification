"""
train_age_gender_model.py
==========================
Multi-task model: predicts BOTH gender and age simultaneously.

Architecture:
  - Shared backbone: MobileNetV2 (pretrained ImageNet)
  - Shared neck:     GlobalAveragePooling2D → Dense(256) → Dropout
  - Head 1 (gender): Dense(1, sigmoid)  → binary 0/1
  - Head 2 (age):    Dense(1, linear)   → continuous 0–116

Why multi-task?
  - Both tasks learn from the same face features
  - Age and gender share visual cues (bone structure, skin texture)
  - Training together acts as a regularizer — harder to overfit
  - One model, one inference call, two outputs

UTKFace filename format:
  [age]_[gender]_[race]_[timestamp].jpg
  Example: 25_0_2_20170116174525125.jpg
  age=25, gender=0 (male)

Run:
    python train_age_gender_model.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

import albumentations as A

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_PATH     = "data/UTKFace"
IMG_SIZE         = 224
BATCH_SIZE       = 64              # use 32 if Colab runs out of RAM
EPOCHS_FROZEN    = 3
EPOCHS_FINETUNE  = 8
MODEL_SAVE_PATH  = "models/age_gender_model.h5"

# Loss weights — how much each task contributes to total loss.
# Age MAE is on a 0-116 scale, gender loss is 0-1 scale.
# Without weighting, age loss dominates and gender suffers.
# 1.0 gender : 0.05 age balances the gradients.
GENDER_LOSS_WEIGHT = 1.0
AGE_LOSS_WEIGHT    = 0.05

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.RandomShadow(shadow_roi=(0,0,1,1), num_shadows_lower=1, num_shadows_upper=2,
                   shadow_dimension=5, p=0.3),
    A.CoarseDropout(max_holes=4, max_height=40, max_width=40,
                    min_holes=1, min_height=10, min_width=10, fill_value=0, p=0.3),
])

# ─────────────────────────────────────────────
# CUSTOM GENERATOR
# ─────────────────────────────────────────────
# Multi-task generator yields (X, {'gender': y_g, 'age': y_a})
# Keras needs named outputs to match the model's output layer names.

class MultiTaskGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, y_gender, y_age, batch_size, transform=None, shuffle=True):
        self.X         = X
        self.y_gender  = y_gender
        self.y_age     = y_age
        self.batch_size = batch_size
        self.transform  = transform
        self.shuffle    = shuffle
        self.indices    = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_X = []

        for i in batch_idx:
            img_uint8 = np.clip(
                (self.X[i] + 1.0) / 2.0 * 255.0, 0, 255
            ).astype(np.uint8)
            if self.transform:
                img_uint8 = self.transform(image=img_uint8)['image']
            batch_X.append(preprocess_input(img_uint8.astype(np.float32)))

        return (
            np.array(batch_X, dtype=np.float32),
            {
                'gender': self.y_gender[batch_idx].astype(np.float32),
                'age':    self.y_age[batch_idx].astype(np.float32)
            }
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

print("\n[1/5] Loading UTKFace...")

images   = []
genders  = []
ages     = []

for filename in tqdm(os.listdir(DATASET_PATH)):
    try:
        parts = filename.split('_')
        if len(parts) < 3:
            continue

        age    = int(parts[0])
        gender = int(parts[1])

        # Skip noise
        if gender not in [0, 1]:
            continue
        if age < 0 or age > 116:
            continue

        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = preprocess_input(image.astype(np.float32))

        images.append(image)
        genders.append(gender)
        ages.append(age)

    except Exception:
        continue

X        = np.array(images,  dtype=np.float32)
y_gender = np.array(genders, dtype=np.float32)
y_age    = np.array(ages,    dtype=np.float32)

print(f"  Total: {len(X)}")
print(f"  Male: {(y_gender==0).sum()} | Female: {(y_gender==1).sum()}")
print(f"  Age range: {y_age.min():.0f} – {y_age.max():.0f} | Mean: {y_age.mean():.1f}")

# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

print("\n[2/5] Splitting...")

idx = np.arange(len(X))
idx_train, idx_test = train_test_split(
    idx, test_size=0.2, random_state=42,
    stratify=y_gender   # stratify by gender to keep balance
)

X_train, X_test       = X[idx_train],        X[idx_test]
yg_train, yg_test     = y_gender[idx_train],  y_gender[idx_test]
ya_train, ya_test     = y_age[idx_train],     y_age[idx_test]

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────

print("\n[3/5] Building generators...")

train_gen = MultiTaskGenerator(X_train, yg_train, ya_train,
                                BATCH_SIZE, train_transform, shuffle=True)
val_gen   = MultiTaskGenerator(X_test,  yg_test,  ya_test,
                                BATCH_SIZE, None, shuffle=False)

# ─────────────────────────────────────────────
# BUILD MULTI-TASK MODEL
# ─────────────────────────────────────────────

print("\n[4/5] Building model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

# ── Shared neck ───────────────────────────────
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x      = base_model(inputs, training=False)
x      = GlobalAveragePooling2D()(x)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.4)(x)

# ── Gender head ───────────────────────────────
# Binary classification → sigmoid → 0 (male) or 1 (female)
gender_out = Dense(1, activation='sigmoid', name='gender')(x)

# ── Age head ──────────────────────────────────
# Regression → linear activation → raw years (0–116)
# No sigmoid/relu on output — we want unbounded regression.
age_out = Dense(1, activation='linear', name='age')(x)

model = Model(inputs=inputs, outputs=[gender_out, age_out])

# Class weights for gender imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=yg_train
)
print(f"  Gender class weights: {dict(enumerate(class_weights))}")

# ─────────────────────────────────────────────
# PHASE 1 — Train heads only (frozen backbone)
# ─────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={
        'gender': 'binary_crossentropy',
        'age':    'mae'                     # Mean Absolute Error for age
    },
    loss_weights={
        'gender': GENDER_LOSS_WEIGHT,
        'age':    AGE_LOSS_WEIGHT
    },
    metrics={
        'gender': 'accuracy',
        'age':    'mae'
    }
)

print(f"\n  ── Phase 1: Head training ({EPOCHS_FROZEN} epochs) ──")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FROZEN,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True,
                      monitor='val_gender_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_loss')
    ]
)

# ─────────────────────────────────────────────
# PHASE 2 — Fine-tune last 30 backbone layers
# ─────────────────────────────────────────────

print(f"\n  ── Phase 2: Fine-tuning ({EPOCHS_FINETUNE} epochs) ──")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        'gender': 'binary_crossentropy',
        'age':    'mae'
    },
    loss_weights={
        'gender': GENDER_LOSS_WEIGHT,
        'age':    AGE_LOSS_WEIGHT
    },
    metrics={
        'gender': 'accuracy',
        'age':    'mae'
    }
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True,
                      monitor='val_gender_accuracy'),
        ReduceLROnPlateau(factor=0.3, patience=3, monitor='val_loss'),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_gender_accuracy',
            verbose=1
        )
    ]
)

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

print("\n[5/5] Evaluation...")

results = model.evaluate(
    np.array([preprocess_input(
        np.clip((X_test[i]+1)/2*255,0,255).astype(np.float32)
    ) for i in range(len(X_test))]),
    {'gender': yg_test, 'age': ya_test},
    verbose=0
)

print(f"  Gender accuracy: {results[3]*100:.2f}%")
print(f"  Age MAE:         {results[4]:.1f} years")
print(f"\nModel saved: {MODEL_SAVE_PATH}")