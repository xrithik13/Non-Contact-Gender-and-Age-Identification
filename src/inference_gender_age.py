"""
inference_gender_age.py
========================
Live webcam inference — predicts Gender + Age for every detected face.

Features carried over from your v2 gender model:
  - RetinaFace detection (every N frames)
  - Face alignment via landmarks
  - CLAHE contrast enhancement
  - Cropping with margin
  - EMA temporal smoothing per tracked face
  - IoU-based face tracking
  - Bias correction for gender

New additions:
  - Age prediction (regression head)
  - Age smoothed separately with its own EMA buffer
  - Display: gender label + confidence + age estimate
"""

import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH             = "models/gender_age_model.h5"
IMG_SIZE               = 224
MAX_AGE                = 116.0      # same value used during training
CONFIDENCE_THRESH      = 0.60       # gender confidence below this → "Uncertain"
DETECT_EVERY_N         = 3
EMA_ALPHA              = 0.20       # lower = smoother (less reactive)
IOU_THRESH             = 0.30
CROP_MARGIN            = 0.10
BIAS_CORRECTION_OFFSET = 0.08      # nudge toward Female; set 0.0 to disable

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("[1/3] Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("  Done.")

# ─────────────────────────────────────────────
# LOAD RETINAFACE
# ─────────────────────────────────────────────

print("[2/3] Loading RetinaFace...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))
print("  Done.")

# ─────────────────────────────────────────────
# CLAHE
# ─────────────────────────────────────────────

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────
# FACE ALIGNMENT
# ─────────────────────────────────────────────

def align_face(image, landmarks):
    try:
        left_eye  = landmarks[0]
        right_eye = landmarks[1]
        dx    = right_eye[0] - left_eye[0]
        dy    = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2)
        )
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = image.shape[:2]
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    except Exception:
        return image

# ─────────────────────────────────────────────
# CROP WITH MARGIN
# ─────────────────────────────────────────────

def crop_with_margin(frame, x1, y1, x2, y2, margin=CROP_MARGIN):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x  = int(bw * margin)
    pad_y  = int(bh * margin)
    x1m = max(0, x1 - pad_x);  y1m = max(0, y1 - pad_y)
    x2m = min(w, x2 + pad_x);  y2m = min(h, y2 + pad_y)
    return frame[y1m:y2m, x1m:x2m]

# ─────────────────────────────────────────────
# PREPROCESS FOR MODEL
# ─────────────────────────────────────────────

def preprocess_face(face_crop):
    face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    processed = preprocess_input(resized.astype(np.float32))
    return np.expand_dims(processed, axis=0)

# ─────────────────────────────────────────────
# IoU — face tracking
# ─────────────────────────────────────────────

def compute_iou(a, b):
    xA = max(a[0], b[0]);  yA = max(a[1], b[1])
    xB = min(a[2], b[2]);  yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(areaA + areaB - inter)

# ─────────────────────────────────────────────
# FACE TRACKER — EMA for both gender and age
# ─────────────────────────────────────────────
#
# Each track stores two EMA values:
#   ema_gender : smoothed gender probability (0=Male, 1=Female)
#   ema_age    : smoothed age prediction (in years)
#
# EMA formula:  ema_new = alpha * raw + (1 - alpha) * ema_old
# With alpha=0.20, it takes ~5 consistent frames to shift the label.
# This eliminates single-frame flicker for both outputs.

class FaceTracker:

    def __init__(self):
        self.tracks  = {}
        self.next_id = 0

    def update(self, detections):
        if not detections:
            self.tracks = {}
            return []

        unmatched = list(range(len(detections)))
        matched   = {}

        for tid, track in self.tracks.items():
            best_iou = IOU_THRESH
            best_idx = None
            for di in unmatched:
                iou = compute_iou(track['box'], detections[di]['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = di
            if best_idx is not None:
                matched[tid] = best_idx
                unmatched.remove(best_idx)

        # Update matched tracks
        for tid, di in matched.items():
            raw_gender = detections[di]['prob_gender']
            raw_age    = detections[di]['pred_age']

            old_ema_g  = self.tracks[tid]['ema_gender']
            old_ema_a  = self.tracks[tid]['ema_age']

            new_ema_g  = EMA_ALPHA * raw_gender + (1 - EMA_ALPHA) * old_ema_g
            new_ema_a  = EMA_ALPHA * raw_age    + (1 - EMA_ALPHA) * old_ema_a

            self.tracks[tid].update({
                'ema_gender': new_ema_g,
                'ema_age':    new_ema_a,
                'box':        detections[di]['box']
            })

            detections[di]['smooth_gender'] = new_ema_g
            detections[di]['smooth_age']    = new_ema_a
            detections[di]['face_id']       = tid

        # New tracks for unmatched detections
        for di in unmatched:
            raw_gender = detections[di]['prob_gender']
            raw_age    = detections[di]['pred_age']
            new_id     = self.next_id
            self.next_id += 1
            self.tracks[new_id] = {
                'box':        detections[di]['box'],
                'ema_gender': raw_gender,
                'ema_age':    raw_age
            }
            detections[di]['smooth_gender'] = raw_gender
            detections[di]['smooth_age']    = raw_age
            detections[di]['face_id']       = new_id

        # Clean up stale tracks
        active_ids = {detections[di]['face_id']
                      for di in range(len(detections))
                      if 'face_id' in detections[di]}
        active_ids.update(matched.keys())
        for tid in list(self.tracks.keys()):
            if tid not in active_ids:
                del self.tracks[tid]

        return detections


tracker = FaceTracker()

# ─────────────────────────────────────────────
# OPEN WEBCAM
# ─────────────────────────────────────────────

print("[3/3] Opening webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Running. Press 'q' to quit.\n")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

frame_count = 0
raw_faces   = []
prev_time   = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_time   = time.time()
    fps         = 1.0 / (curr_time - prev_time + 1e-9)
    prev_time   = curr_time

    enhanced = apply_clahe(frame)

    # Run RetinaFace every N frames (saves CPU)
    if frame_count % DETECT_EVERY_N == 0:
        raw_faces = app.get(enhanced)

    detections = []

    for face in raw_faces:
        box       = face.bbox.astype(int)
        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Align face using eye landmarks
        if hasattr(face, 'kps') and face.kps is not None:
            aligned = align_face(enhanced, face.kps)
        else:
            aligned = enhanced

        face_crop = crop_with_margin(aligned, x1, y1, x2, y2)
        if face_crop.size == 0:
            continue

        # Run model — returns [gender_output, age_output]
        input_tensor = preprocess_face(face_crop)
        preds        = model.predict(input_tensor, verbose=0)

        raw_gender_prob = float(preds[0][0][0])
        raw_age_years   = float(preds[1][0][0]) * MAX_AGE   # de-normalise

        # Clamp age to sane range (model can sometimes go slightly outside)
        raw_age_years = float(np.clip(raw_age_years, 1, MAX_AGE))

        # Bias correction on gender — applied before smoothing
        corrected_gender = min(1.0, raw_gender_prob + BIAS_CORRECTION_OFFSET)

        detections.append({
            'box':        [x1, y1, x2, y2],
            'prob_gender': corrected_gender,
            'pred_age':    raw_age_years,
            'raw_gender':  raw_gender_prob
        })

    # Temporal smoothing
    smoothed = tracker.update(detections)

    # ─────────── DRAW ─────────────────────────
    for det in smoothed:
        x1, y1, x2, y2 = det['box']
        smooth_gender   = det['smooth_gender']
        smooth_age      = det['smooth_age']
        raw_gender      = det['raw_gender']

        # ── Gender ──
        if smooth_gender > 0.5:
            gender     = "Female"
            confidence = smooth_gender
            box_color  = (255, 105, 210)    # pink
        else:
            gender     = "Male"
            confidence = 1.0 - smooth_gender
            box_color  = (100, 220, 100)    # green

        confidence_pct = confidence * 100

        if confidence < CONFIDENCE_THRESH:
            gender_label = f"Uncertain ({confidence_pct:.0f}%)"
            label_color  = (0, 165, 255)
        else:
            gender_label = f"{gender} {confidence_pct:.0f}%"
            label_color  = box_color

        # ── Age ──
        age_label = f"Age ~{smooth_age:.0f}"

        # ── Box ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # ── Gender label bar (above box) ──
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.60
        thickness  = 2

        (gw, gh), _ = cv2.getTextSize(gender_label, font, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - gh - 12), (x1 + gw + 8, y1), label_color, -1)
        cv2.putText(frame, gender_label, (x1 + 4, y1 - 6),
                    font, font_scale, (0, 0, 0), thickness)

        # ── Age label bar (below top label) ──
        (aw, ah), _ = cv2.getTextSize(age_label, font, 0.55, 1)
        age_y_top  = y1 - gh - 12 - ah - 8
        age_y_bot  = y1 - gh - 12

        # Fall back: draw age BELOW box if there's no room above
        if age_y_top < 0:
            age_y = y2 + ah + 6
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + aw + 8, y2 + ah + 10),
                          (50, 50, 50), -1)
            cv2.putText(frame, age_label, (x1 + 4, age_y),
                        font, 0.55, (220, 220, 220), 1)
        else:
            cv2.rectangle(frame, (x1, age_y_top - 2), (x1 + aw + 8, age_y_bot),
                          (40, 40, 40), -1)
            cv2.putText(frame, age_label, (x1 + 4, age_y_bot - 4),
                        font, 0.55, (220, 220, 220), 1)

        # ── Debug line ──
        debug = f"g_raw:{raw_gender:.2f} g_sm:{smooth_gender:.2f} age_sm:{smooth_age:.1f}"
        cv2.putText(frame, debug, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)

    # ── HUD overlays ──
    cv2.putText(frame, f"FPS: {fps:.1f}",          (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
    cv2.putText(frame, f"Faces: {len(smoothed)}",  (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
    cv2.putText(frame, f"Bias: +{BIAS_CORRECTION_OFFSET:.2f}", (10, 71),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    cv2.imshow("Gender + Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
