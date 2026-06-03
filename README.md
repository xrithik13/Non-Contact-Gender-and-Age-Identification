# Non-Contact Gender and Age Identification Using Cameras

## Overview

This project implements a real-time, non-contact gender and age identification system using computer vision and deep learning.

The system captures live video from a webcam, detects faces using RetinaFace, and predicts both gender and age using a MobileNetV2-based multi-task deep learning model.

The solution is designed to be lightweight, CPU-friendly, and suitable for real-time applications.

---

## Features

* Real-time webcam inference
* RetinaFace face detection
* Face alignment using facial landmarks
* Gender classification (Male / Female)
* Age estimation
* Temporal smoothing to reduce prediction flicker
* CPU-optimized inference
* MobileNetV2 transfer learning

---

## Project Pipeline

Webcam Feed

↓

RetinaFace Face Detection

↓

Face Alignment

↓

Face Cropping & Preprocessing

↓

MobileNetV2 Multi-Task Network

↓

Gender Prediction + Age Estimation

---

## Dataset

UTKFace Dataset

Filename format:

age_gender_race_timestamp.jpg

Example:

25_0_2_20170116174525125.jpg

Where:

* Age = 25
* Gender = 0 (Male)
* Race = 2

---

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* MobileNetV2
* InsightFace (RetinaFace)
* NumPy
* Scikit-Learn

---

## Project Structure

gender_detection/

├── data/

├── models/

├── outputs/

├── src/

│ ├── prepare_dataset.py

│ ├── train_gender_age_model.py

│ ├── inference_gender_age.py

│ ├── test_retinaface.py

│ ├── test_mobilenet.py

│ └── test_camera.py

├── requirements.txt

└── README.md

---

## Current Status

✅ Environment setup completed

✅ Dataset preparation completed

✅ RetinaFace integration completed

✅ MobileNetV2 architecture implemented

✅ Gender + Age model training pipeline completed

✅ Real-time webcam inference completed

🚧 Performance optimization in progress

---

## Research Objectives

* Non-contact gender identification
* Real-time age estimation
* Lightweight deployment on CPU systems
* Robust performance under varying lighting conditions
* Face alignment and preprocessing improvements
* Real-time inference optimization

---

## Future Improvements

* TensorFlow Lite deployment
* Edge-device optimization
* Model quantization
* Fairness analysis across demographic groups
* Multi-person tracking improvements
* Web-based deployment using Flask or FastAPI

---

## Author

Rithik Roshan V

