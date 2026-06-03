"""
train_gender_age_model.py
==========================
Multi-task MobileNetV2 — predicts BOTH gender and age
from a single shared backbone.

Architecture:
    MobileNetV2 (frozen → then partially unfrozen)
        └─ GlobalAveragePooling2D
            ├─ Dense(256) → Dropout → gender_output  (sigmoid, binary)
            └─ Dense(256) → Dropout → age_output     (linear, regression)

Loss:
    Total = binary_crossentropy(gender) + age_weight * mae(age)
    age_weight = 2.0  — upweights age loss so it isn't ignored

Labels:
    gender : 0 = Male, 1 = Female
    age    : normalised to [0,1]  →  multiply by 116 to get real age

Run:
    python train_gender_age_model.py
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
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_PATH    = "data/UTKFace"
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 5       # Phase 1 — train only custom head
EPOCHS_FINETUNE = 20      # Phase 2 — unfreeze last 30 backbone layers
MODEL_SAVE_PATH = "models/gender_age_model.h5"
MAX_AGE         = 116.0   # for normalisation

AGE_LOSS_WEIGHT = 2.0     # weight applied to age MAE in total loss
                           # increase → model tries harder on age

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# TQDM EPOCH PROGRESS CALLBACK
# ─────────────────────────────────────────────
# Keras default progress bar is per-step.
# This shows a clean tqdm bar per epoch instead,
# printing a one-line summary after each epoch completes.

class TQDMEpochCallback(Callback):

    def __init__(self, total_epochs, phase_name="Training"):
        super().__init__()
        self.total_epochs = total_epochs
        self.phase_name   = phase_name
        self.pbar         = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(
            total=self.total_epochs,
            desc=f"  {self.phase_name}",
            unit="epoch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        g_acc   = logs.get('gender_output_accuracy',     0)
        vg_acc  = logs.get('val_gender_output_accuracy', 0)
        a_mae   = logs.get('age_output_mae',             0) * MAX_AGE
        va_mae  = logs.get('val_age_output_mae',         0) * MAX_AGE
        loss    = logs.get('loss',     0)
        val_loss= logs.get('val_loss', 0)

        self.pbar.set_postfix({
            'loss'    : f"{loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'g_acc'   : f"{g_acc*100:.1f}%",
            'vg_acc'  : f"{vg_acc*100:.1f}%",
            'age_mae' : f"{a_mae:.1f}yr",
            'vage_mae': f"{va_mae:.1f}yr",
        }, refresh=True)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

print("\n[1/5] Loading UTKFace dataset...")

images, gender_labels, age_labels = [], [], []

filenames = os.listdir(DATASET_PATH)

for filename in tqdm(filenames, desc="  Reading images", unit="img", dynamic_ncols=True):
    try:
        parts = filename.split('_')
        if len(parts) < 2:
            continue

        age    = int(parts[0])
        gender = int(parts[1])

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

        # MobileNetV2 normalisation → scales pixels to [-1, 1]
        image = preprocess_input(image.astype(np.float32))

        images.append(image)
        gender_labels.append(float(gender))
        age_labels.append(age / MAX_AGE)       # normalise to [0,1]

    except Exception:
        continue

X        = np.array(images,        dtype=np.float32)
y_gender = np.array(gender_labels, dtype=np.float32)
y_age    = np.array(age_labels,    dtype=np.float32)

print(f"\n  Total  : {X.shape[0]} images")
print(f"  Male   : {int((y_gender==0).sum())}  |  Female: {int((y_gender==1).sum())}")
print(f"  Age    : min={y_age.min()*MAX_AGE:.0f}  "
      f"max={y_age.max()*MAX_AGE:.0f}  "
      f"mean={y_age.mean()*MAX_AGE:.1f}")

# ─────────────────────────────────────────────
# STEP 2 — TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

print("\n[2/5] Splitting dataset...")

X_train, X_test, \
yg_train, yg_test, \
ya_train, ya_test = train_test_split(
    X, y_gender, y_age,
    test_size=0.2,
    random_state=42,
    stratify=y_gender      # keep male/female ratio equal in both splits
)

print(f"  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]}  samples")

# ─────────────────────────────────────────────
# STEP 3 — DATA AUGMENTATION
# ─────────────────────────────────────────────
# Multi-output generators need a custom generator because
# ImageDataGenerator.flow() supports only one label array.
# We wrap it to yield (X, {'gender_output': y_g, 'age_output': y_a}).

print("\n[3/5] Setting up data augmentation...")

augmentor = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

def multi_output_generator(X, y_gender, y_age, batch_size, augment=True):
    """
    Yields (images, {'gender_output': gender_batch, 'age_output': age_batch})
    Augments images when augment=True.
    """
    n       = len(X)
    indices = np.arange(n)

    while True:
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end       = min(start + batch_size, n)
            batch_idx = indices[start:end]

            X_batch  = X[batch_idx].copy()
            yg_batch = y_gender[batch_idx]
            ya_batch = y_age[batch_idx]

            if augment:
                aug_imgs = []
                for img in X_batch:
                    aug_imgs.append(augmentor.random_transform(img))
                X_batch = np.array(aug_imgs, dtype=np.float32)

            yield X_batch, {
                'gender_output': yg_batch,
                'age_output':    ya_batch
            }

train_gen = multi_output_generator(X_train, yg_train, ya_train, BATCH_SIZE, augment=True)
val_gen   = multi_output_generator(X_test,  yg_test,  ya_test,  BATCH_SIZE, augment=False)

steps_per_epoch  = len(X_train) // BATCH_SIZE
validation_steps = len(X_test)  // BATCH_SIZE

print(f"  Steps/epoch : {steps_per_epoch}")
print(f"  Val steps   : {validation_steps}")

# ─────────────────────────────────────────────
# STEP 4 — BUILD MULTI-TASK MODEL
# ─────────────────────────────────────────────

print("\n[4/5] Building multi-task model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

# ── Shared backbone ──────────────────────────
inputs   = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
backbone = base_model(inputs, training=False)
shared   = GlobalAveragePooling2D()(backbone)

# ── Gender head (binary classification) ──────
g = Dense(256, activation='relu')(shared)
g = Dropout(0.4)(g)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(g)

# ── Age head (regression) ─────────────────────
# linear activation — output ∈ [0,1] (normalised age)
# MAE loss is better than MSE for age: less sensitive to outliers
a = Dense(256, activation='relu')(shared)
a = Dropout(0.4)(a)
age_output = Dense(1, activation='linear', name='age_output')(a)

model = Model(inputs=inputs, outputs=[gender_output, age_output])

# ── Class weights for gender imbalance ────────
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(yg_train),
    y=yg_train
)
class_weight_dict = dict(enumerate(class_weights_arr))
print(f"  Gender class weights: {class_weight_dict}")

model.summary()

# ─────────────────────────────────────────────
# PHASE 1 — Train head only (backbone frozen)
# ─────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'gender_output': 'binary_crossentropy',
        'age_output':    'mae'
    },
    loss_weights={
        'gender_output': 1.0,
        'age_output':    AGE_LOSS_WEIGHT
    },
    metrics={
        'gender_output': 'accuracy',
        'age_output':    'mae'
    }
)

# Verify exact metric names so monitor strings are correct
dummy_gen  = multi_output_generator(X_train[:BATCH_SIZE], yg_train[:BATCH_SIZE],
                                    ya_train[:BATCH_SIZE], BATCH_SIZE, augment=False)
dummy_hist = model.fit(next(dummy_gen)[0],
                       next(dummy_gen)[1],
                       epochs=1, verbose=0)
metric_names = list(dummy_hist.history.keys())
print(f"\n  Detected metric names: {metric_names}")

# Pick the correct validation accuracy metric name
gender_acc_metric = 'val_gender_output_accuracy'
if 'val_gender_output_accuracy' not in [
    k for k in ['val_gender_output_accuracy',
                'val_gender_accuracy',
                'val_gender_output_acc']
]:
    # Fallback: find it from dummy run
    for k in metric_names:
        if 'gender' in k and 'acc' in k:
            gender_acc_metric = 'val_' + k if not k.startswith('val_') else k
            break

print(f"  Using monitor: '{gender_acc_metric}'")

callbacks_phase1 = [
    EarlyStopping(
        patience=3,
        restore_best_weights=True,
        monitor='val_gender_output_accuracy',
        mode='max'                             # ← FIX: must specify mode for custom metric names
    ),
    ReduceLROnPlateau(
        factor=0.5,
        patience=2,
        monitor='val_loss',
        mode='min',
        verbose=0
    ),
    TQDMEpochCallback(total_epochs=EPOCHS_FROZEN, phase_name="Phase 1 (frozen)")
]

print(f"\n  ── Phase 1: Training head only ({EPOCHS_FROZEN} epochs) ──")
model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_phase1,
    verbose=0          # ← silence Keras default bar; tqdm handles display
)

# ─────────────────────────────────────────────
# PHASE 2 — Fine-tune last 30 backbone layers
# ─────────────────────────────────────────────
# Unfreezing top layers lets the backbone specialise
# away from generic ImageNet patterns toward facial geometry.
# Lower LR prevents destroying the already-learned weights.

print(f"\n  ── Phase 2: Fine-tuning last 30 layers ({EPOCHS_FINETUNE} epochs) ──")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"  Backbone layers unfrozen: {trainable_count} / {len(base_model.layers)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'gender_output': 'binary_crossentropy',
        'age_output':    'mae'
    },
    loss_weights={
        'gender_output': 1.0,
        'age_output':    AGE_LOSS_WEIGHT
    },
    metrics={
        'gender_output': 'accuracy',
        'age_output':    'mae'
    }
)

callbacks_phase2 = [
    EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_gender_output_accuracy',
        mode='max'                             # ← FIX
    ),
    ReduceLROnPlateau(
        factor=0.3,
        patience=3,
        monitor='val_loss',
        mode='min',
        verbose=0
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_gender_output_accuracy',
        mode='max',                            # ← FIX
        verbose=1
    ),
    TQDMEpochCallback(total_epochs=EPOCHS_FINETUNE, phase_name="Phase 2 (finetune)")
]

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks_phase2,
    verbose=0          # ← silence Keras default bar
)

# ─────────────────────────────────────────────
# STEP 5 — EVALUATE
# ─────────────────────────────────────────────

print("\n[5/5] Final Evaluation on test set...")

eval_targets = {'gender_output': yg_test, 'age_output': ya_test}
results      = model.evaluate(X_test, eval_targets, verbose=0)

# Print all metrics so indices are clear
print("\n  All evaluation metrics:")
for name, val in zip(model.metrics_names, results):
    print(f"    {name:40s} : {val:.4f}")

# Human-readable summary
# Safe metric extraction by name
metrics_dict = dict(zip(model.metrics_names, results))
g_acc  = metrics_dict.get('gender_output_accuracy', 0)
a_mae  = metrics_dict.get('age_output_mae', 0) * MAX_AGE

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │  Gender Accuracy : {g_acc*100:6.2f}%       │")
print(f"  │  Age MAE         : {a_mae:6.1f} years   │")
print(f"  └─────────────────────────────────┘")

# Quick sanity check — 5 sample predictions
print("\n  Sample predictions (first 5 test images):")
print(f"  {'#':<4} {'True Gender':<14} {'Pred Gender':<14} {'Conf':<8} {'True Age':<10} {'Pred Age'}")
print(f"  {'-'*64}")

sample_preds = model.predict(X_test[:5], verbose=0)
for i in range(5):
    g_pred    = float(sample_preds[0][i][0])
    a_pred    = float(sample_preds[1][i][0]) * MAX_AGE
    a_pred    = float(np.clip(a_pred, 1, MAX_AGE))
    g_true    = int(yg_test[i])
    a_true    = ya_test[i] * MAX_AGE
    g_label   = "Female" if g_pred > 0.5 else "Male"
    g_true_l  = "Female" if g_true == 1  else "Male"
    conf      = g_pred if g_pred > 0.5 else 1 - g_pred
    match     = "✓" if g_label == g_true_l else "✗"
    print(f"  {i:<4} {g_true_l:<14} {g_label:<14} {conf*100:5.1f}%   "
          f"{a_true:5.0f} yrs   {a_pred:5.1f} yrs  {match}")

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print("Done.")
