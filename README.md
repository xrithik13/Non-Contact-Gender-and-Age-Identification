# Non-Contact Gender Estimation System

Camera-based gender classification using MTCNN face detection + MobileNetV2 classifier.

---

## Project structure

```
gender_estimation/
├── data/
│   ├── raw/
│   │   └── UTKFace/          ← place dataset here
│   └── processed/
├── models/
│   ├── checkpoints/          ← saved during training
│   └── saved/                ← final model (.h5)
├── results/                  ← plots, metrics, saved frames
├── src/
│   ├── config.py             ← all paths and hyperparameters
│   ├── dataset.py            ← data loading, augmentation, splits
│   ├── face_detector.py      ← MTCNN wrapper
│   ├── model.py              ← MobileNetV2 classifier
│   ├── train.py              ← training script
│   ├── evaluate.py           ← test set metrics + fairness
│   └── inference.py          ← live webcam demo
└── requirements.txt
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download **UTKFace** from:
https://susanqq.github.io/UTKFace/

Extract and place at:
```
data/raw/UTKFace/
```

Files should look like:
```
data/raw/UTKFace/25_0_0_201701....jpg
data/raw/UTKFace/32_1_2_201706....jpg
```

---

## Run

### Train the model

```bash
cd gender_estimation
python src/train.py
```

This runs two phases:
- **Phase 1 (warm-up):** trains only the classification head for 10 epochs.
- **Phase 2 (fine-tune):** unfreezes top backbone layers and trains for up to 30 epochs.

Training curves are saved to `results/`.

### Evaluate on test set

```bash
python src/evaluate.py --model models/saved/gender_model.h5
```

Outputs: confusion matrix, per-race fairness breakdown.

### Real-time webcam inference

```bash
python src/inference.py --model models/saved/gender_model.h5 --camera 0
```

- `Q` → quit
- `S` → save current frame

---

## Key design choices

| Choice | Reason |
|---|---|
| MTCNN | Best balance of speed and accuracy for multi-face detection |
| MobileNetV2 backbone | Lightweight, fast inference on CPU, high accuracy |
| Two-phase training | Avoids catastrophic forgetting of ImageNet features |
| UTKFace dataset | Age + gender + race labels → enables fairness analysis |
| Stratified split | Preserves class balance across train/val/test |
| Augmentation | Flip, brightness, crop, occlusion patch for robustness |

---

## Expected performance (UTKFace)

| Metric | Typical result |
|---|---|
| Test accuracy | 92–95% |
| ROC-AUC | 0.97–0.99 |

---

## Next steps

- Export to TensorFlow Lite for mobile/edge deployment
- Add age estimation as a secondary head (multi-task learning)
- Integrate with a Raspberry Pi + Pi Camera for embedded deployment
