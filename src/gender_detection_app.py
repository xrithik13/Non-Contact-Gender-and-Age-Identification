import cv2
import numpy as np
import time
from collections import deque

from insightface.app import FaceAnalysis
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH             = "models/gender_model_v2.h5"
IMG_SIZE               = 224
CONFIDENCE_THRESH      = 0.60
DETECT_EVERY_N         = 3
SMOOTH_WINDOW          = 12
EMA_ALPHA              = 0.20
AGE_EMA_ALPHA          = 0.12    # age smoothing — slower than gender
IOU_THRESH             = 0.30
CROP_MARGIN            = 0.10
BIAS_CORRECTION_OFFSET = 0.08

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("[1/3] Loading model...")
model = load_model(MODEL_PATH)
print("  Done.")

# ─────────────────────────────────────────────
# LOAD RETINAFACE
# buffalo_l has built-in age estimator → face.age
# ─────────────────────────────────────────────

print("[2/3] Loading RetinaFace...")
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)
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
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    x1m = max(0, x1 - pad_x)
    y1m = max(0, y1 - pad_y)
    x2m = min(w, x2 + pad_x)
    y2m = min(h, y2 + pad_y)
    return frame[y1m:y2m, x1m:x2m]

# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def preprocess_face(face_crop):
    face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    processed = preprocess_input(resized.astype(np.float32))
    return np.expand_dims(processed, axis=0)

# ─────────────────────────────────────────────
# IoU
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
# FACE TRACKER — EMA for gender + age
# ─────────────────────────────────────────────

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

        for tid, di in matched.items():
            # gender EMA
            raw     = detections[di]['prob']
            old_ema = self.tracks[tid]['ema']
            new_ema = EMA_ALPHA * raw + (1 - EMA_ALPHA) * old_ema
            # age EMA
            age_raw     = detections[di]['age_raw']
            old_age_ema = self.tracks[tid]['age_ema']
            new_age_ema = AGE_EMA_ALPHA * age_raw + (1 - AGE_EMA_ALPHA) * old_age_ema

            self.tracks[tid]['ema']     = new_ema
            self.tracks[tid]['age_ema'] = new_age_ema
            self.tracks[tid]['box']     = detections[di]['box']
            detections[di]['smooth_prob'] = new_ema
            detections[di]['smooth_age']  = new_age_ema
            detections[di]['face_id']     = tid

        for di in unmatched:
            prob    = detections[di]['prob']
            age_raw = detections[di]['age_raw']
            new_id  = self.next_id
            self.next_id += 1
            self.tracks[new_id] = {
                'box':     detections[di]['box'],
                'ema':     prob,
                'age_ema': age_raw
            }
            detections[di]['smooth_prob'] = prob
            detections[di]['smooth_age']  = age_raw
            detections[di]['face_id']     = new_id

        active = set(matched.keys())
        for tid in list(self.tracks.keys()):
            if tid not in active and tid not in [
                detections[di]['face_id']
                for di in range(len(detections))
                if 'face_id' in detections[di]
            ]:
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

frame_count = 0
raw_faces   = []
prev_time   = time.time()

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-9)
    prev_time = curr_time

    enhanced = apply_clahe(frame)

    if frame_count % DETECT_EVERY_N == 0:
        raw_faces = app.get(enhanced)

    detections = []

    for face in raw_faces:
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        if hasattr(face, 'kps') and face.kps is not None:
            aligned = align_face(enhanced, face.kps)
        else:
            aligned = enhanced

        face_crop = crop_with_margin(aligned, x1, y1, x2, y2)
        if face_crop.size == 0:
            continue

        # gender prediction — our MobileNet
        input_tensor   = preprocess_face(face_crop)
        prediction     = model.predict(input_tensor, verbose=0)
        raw_prob       = float(prediction[0][0])
        corrected_prob = min(1.0, raw_prob + BIAS_CORRECTION_OFFSET)

        # age — insightface buffalo_l built-in
        age_raw = float(face.age) if hasattr(face, 'age') and face.age is not None else 25.0
        age_raw = float(np.clip(age_raw, 1, 100))

        detections.append({
            'box':      [x1, y1, x2, y2],
            'prob':     corrected_prob,
            'raw_prob': raw_prob,
            'age_raw':  age_raw
        })

    smoothed = tracker.update(detections)

    for det in smoothed:
        x1, y1, x2, y2 = det['box']
        smooth_prob = det['smooth_prob']
        raw_prob    = det['raw_prob']
        age         = int(round(det['smooth_age']))

        if smooth_prob > 0.5:
            gender     = "Female"
            confidence = smooth_prob
            color      = (255, 105, 210)
        else:
            gender     = "Male"
            confidence = 1.0 - smooth_prob
            color      = (100, 255, 100)

        confidence_pct = confidence * 100

        if confidence < CONFIDENCE_THRESH:
            label = f"Uncertain | {age} yrs"
            color = (0, 165, 255)
        else:
            label = f"{gender} | {age} yrs | {confidence_pct:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

        debug = f"raw:{raw_prob:.2f} smooth:{smooth_prob:.2f} age:{det['smooth_age']:.1f}"
        cv2.putText(frame, debug, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Faces: {len(smoothed)}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Gender + Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")