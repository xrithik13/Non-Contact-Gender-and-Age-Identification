# Gender Detection Model - Google Colab GPU Training
# =====================================================
# Run each cell in order. GPU trains ~10x faster than CPU.
#
# SETUP STEPS BEFORE RUNNING:
# 1. Runtime → Change runtime type → T4 GPU → Save
# 2. Upload UTKFace.zip to Colab or use Kaggle API
# 3. Run all cells top to bottom

# ──────────────────────────────────────────────────────
# CELL 1 — Check GPU
# ──────────────────────────────────────────────────────

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Should print something like:
# GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# If it says [] — go to Runtime → Change runtime type → T4 GPU

# ──────────────────────────────────────────────────────
# CELL 2 — Install dependencies
# ──────────────────────────────────────────────────────

# Run this cell once
import subprocess
subprocess.run(["pip", "install", "albumentations", "-q"])
subprocess.run(["pip", "install", "tqdm", "-q"])

print("Dependencies installed.")

# ──────────────────────────────────────────────────────
# CELL 3 — Upload UTKFace dataset
# ──────────────────────────────────────────────────────
# OPTION A: Upload zip manually (slow for large files)
# OPTION B: Kaggle API (recommended — fastest)
#
# For Kaggle API:
# 1. Go to kaggle.com → Account → Create API Token
# 2. Download kaggle.json
# 3. Run the Kaggle cell below

# --- OPTION A: Manual upload ---
# from google.colab import files
# uploaded = files.upload()   # select UTKFace.zip
# !unzip -q UTKFace.zip -d data/

# --- OPTION B: Kaggle API (recommended) ---
# Upload your kaggle.json first:
from google.colab import files
print("Upload your kaggle.json file:")
# uploaded = files.upload()   # uncomment this line and upload kaggle.json

# Then run:
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d jangedoo/utkface-new
# !unzip -q utkface-new.zip -d data/
# !ls data/

# ── After upload, verify ──────────────────────────────
import os

# Adjust this path based on how your zip extracted
DATASET_PATH = "data/UTKFace"   # change if needed

if os.path.exists(DATASET_PATH):
    files_count = len(os.listdir(DATASET_PATH))
    print(f"Dataset found: {files_count} files in {DATASET_PATH}")
else:
    print(f"Dataset NOT found at {DATASET_PATH}")
    print("Available directories:", os.listdir("data/") if os.path.exists("data/") else "data/ not found")

# ──────────────────────────────────────────────────────
# CELL 4 — Imports
# ──────────────────────────────────────────────────────

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

print("All imports successful.")

# ──────────────────────────────────────────────────────
# CELL 5 — Config
# ──────────────────────────────────────────────────────

DATASET_PATH    = "data/UTKFace"   # adjust if needed
IMG_SIZE        = 224
BATCH_SIZE      = 64               # larger batch = faster on GPU (was 32 on CPU)
EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 20               # more epochs — GPU can handle it
MODEL_SAVE_PATH = "gender_model_v3.h5"

os.makedirs("models", exist_ok=True)

print(f"Config ready. IMG_SIZE={IMG_SIZE}, BATCH={BATCH_SIZE}")

# ──────────────────────────────────────────────────────
# CELL 6 — Load dataset
# ──────────────────────────────────────────────────────

print("Loading UTKFace...")

images = []
labels = []

all_files = os.listdir(DATASET_PATH)

for filename in tqdm(all_files):
    try:
        parts = filename.split('_')
        if len(parts) < 2:
            continue

        gender = int(parts[1])
        if gender not in [0, 1]:
            continue

        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = preprocess_input(image.astype(np.float32))

        images.append(image)
        labels.append(gender)

    except Exception:
        continue

X = np.array(images, dtype=np.float32)
y = np.array(labels)

print(f"\nLoaded: {len(X)} images")
print(f"Male (0):   {(y==0).sum()}")
print(f"Female (1): {(y==1).sum()}")

# ──────────────────────────────────────────────────────
# CELL 7 — Train/test split
# ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ──────────────────────────────────────────────────────
# CELL 8 — Albumentations augmentation pipeline
# ──────────────────────────────────────────────────────

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.RandomShadow(shadow_roi=(0,0,1,1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.3),
    A.CoarseDropout(max_holes=4, max_height=40, max_width=40, min_holes=1, min_height=10, min_width=10, fill_value=0, p=0.3),
])

val_transform = A.Compose([])

print("Augmentation pipeline ready.")

# ──────────────────────────────────────────────────────
# CELL 9 — Custom data generator
# ──────────────────────────────────────────────────────

class AlbumentationsGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, y, batch_size, transform=None, shuffle=True):
        self.X          = X
        self.y          = y
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
        batch_y = self.y[batch_idx]

        for i in batch_idx:
            img_uint8 = np.clip((self.X[i] + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            if self.transform:
                augmented = self.transform(image=img_uint8)
                img_uint8 = augmented['image']
            img_processed = preprocess_input(img_uint8.astype(np.float32))
            batch_X.append(img_processed)

        return np.array(batch_X, dtype=np.float32), batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_gen = AlbumentationsGenerator(X_train, y_train, BATCH_SIZE, train_transform, shuffle=True)
val_gen   = AlbumentationsGenerator(X_test,  y_test,  BATCH_SIZE, val_transform,   shuffle=False)

print("Generators ready.")

# ──────────────────────────────────────────────────────
# CELL 10 — Build model
# ──────────────────────────────────────────────────────

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs  = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dense(256, activation='relu')(x)
x       = Dropout(0.4)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
print(f"Model built. Parameters: {model.count_params():,}")

# ──────────────────────────────────────────────────────
# CELL 11 — Phase 1: Train head (frozen backbone)
# ──────────────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\nPhase 1: Training head only ({EPOCHS_FROZEN} epochs)...")

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
    ]
)

# ──────────────────────────────────────────────────────
# CELL 12 — Phase 2: Fine-tune last 30 layers
# ──────────────────────────────────────────────────────

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen layers in backbone: {trainable_count}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\nPhase 2: Fine-tuning ({EPOCHS_FINETUNE} epochs)...")

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
)

# ──────────────────────────────────────────────────────
# CELL 13 — Evaluate
# ──────────────────────────────────────────────────────

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
print(f"Final Test Loss:     {loss:.4f}")

# ──────────────────────────────────────────────────────
# CELL 14 — Plot training curves
# ──────────────────────────────────────────────────────

import matplotlib.pyplot as plt

# Combine both training phases
all_acc     = history1.history['accuracy']     + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss    = history1.history['loss']         + history2.history['loss']
all_val_loss= history1.history['val_loss']     + history2.history['val_loss']

epochs_range = range(1, len(all_acc) + 1)
phase1_end   = len(history1.history['accuracy'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(epochs_range, all_acc,     label='Train Accuracy', color='royalblue')
ax1.plot(epochs_range, all_val_acc, label='Val Accuracy',   color='tomato')
ax1.axvline(x=phase1_end + 0.5, color='gray', linestyle='--', label='Fine-tuning starts')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(epochs_range, all_loss,     label='Train Loss', color='royalblue')
ax2.plot(epochs_range, all_val_loss, label='Val Loss',   color='tomato')
ax2.axvline(x=phase1_end + 0.5, color='gray', linestyle='--', label='Fine-tuning starts')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Gender Model v3 — Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

print("Plot saved as training_curves.png")

# ──────────────────────────────────────────────────────
# CELL 15 — Download model
# ──────────────────────────────────────────────────────
# Downloads gender_model_v3.h5 to your computer.
# Then put it in your local project's models/ folder.

from google.colab import files

files.download(MODEL_SAVE_PATH)
print(f"Downloading {MODEL_SAVE_PATH}...")
print("\nDone! Put this file in your local project's models/ folder.")
print("Then run: python gender_detection_app_v3.py")
