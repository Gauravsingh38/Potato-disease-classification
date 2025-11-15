
#### 📘 Potato Disease Classification Using Deep Learning (CNN)

### *End-to-End System — Data → CNN → FastAPI → Web App → Mobile Ready*

---

## 🧭 Table of Contents

1. Project Overview
2. Problem Context & Motivation
3. Business Use-Case
4. Project Architecture
5. Dataset Overview
6. Data Preprocessing & Augmentation
7. TensorFlow Data Pipeline
8. CNN Model Architecture
9. Training, Evaluation & Results
10. Model Saving & Versioning
11. FastAPI Backend (main-aug.py)
12. Frontend (React.js)
13. Mobile App (Future Integration - TFLite)
14. Deployment Guide
15. Project Folder Structure
16. How to Run the Project
17. Future Improvements
18. Credits

---

# 1. 📌 Project Overview

Potatoes are one of the most widely grown crops worldwide. Their productivity is significantly affected by two major leaf diseases:

- **Early Blight**
- **Late Blight**

Early detection is crucial but often not accessible to small-scale farmers.

This project solves this problem by building an **AI-powered potato disease detection system** using:

✔ Deep Learning (CNN)

✔ TensorFlow/Keras

✔ FastAPI backend

✔ React.js web app

✔ TFLite support for mobile

✔ Docker-ready deployment

The final system allows anyone to upload an image of a potato leaf and instantly get:

- Disease classification
- Confidence score
- Simple UI experience

---

# 2. 🌱 Problem Context & Motivation

Farmers typically rely on **manual inspection** of leaf conditions, which is:

✘ Error-prone

✘ Slow

✘ Requires expertise

Diseases like **Late Blight** can destroy entire potato fields in days.

A simple smartphone-based AI system can:

✔ Improve farmer decision-making

✔ Reduce crop loss

✔ Increase agricultural efficiency

✔ Scale across remote regions

---

# 3. 🏢 Business Use-Case

Developed for *AtliQ Agriculture* as a real-world agritech solution:

### Farmers can:

📸 Capture a potato leaf image

→ Instantly receive prediction (Healthy / Early Blight / Late Blight)

### Organization benefits:

✔ Low-cost scalable tool

✔ Future extension to multiple crops

✔ Can integrate into agritech platforms

---

# 4. 🏗 Project Architecture

```
                ┌──────────────────────┐
                │   PlantVillage Data  │
                └──────────┬───────────┘
                           │
                 Data Preprocessing
                           │
                 CNN Model (main-aug.py)
                           │
               SavedModel + Versioning
                           │
               FastAPI Backend (main-aug)
                           │
             REST API (JSON responses)
                           │
     ┌─────────────────────┴─────────────────────┐
     │                                           │
React Web App                               Mobile App (TFLite)

```

---

# 5. 🗂 Dataset Overview

Dataset used: **PlantVillage (Kaggle)**

Classes retained:

1. **Potato___Healthy**
2. **Potato___Early_Blight**
3. **Potato___Late_Blight**

Each class contains **~1000 images**.

Data structure used:

```
potato_disease/
    ├── Potato___Early_Blight/
    ├── Potato___Late_Blight/
    └── Potato___Healthy/

```

---

# 6. 🧼 Data Preprocessing & Augmentation

## Normalization

All images resized to **256 × 256 × 3**

Scaled to **0–1** using TensorFlow’s `Rescaling(1./255)` layer.

## Augmentation (main-aug model)

Applied using:

- Random Flip (horizontal + vertical)
- Random Rotation (0.2)

Purpose:

✔ Reduce overfitting

✔ Create robust model

✔ Improve generalization

---

# 7. ⚙ TensorFlow Data Pipeline

Built using:

- `tf.data.Dataset`
- `image_dataset_from_directory`
- `cache()`
- `shuffle()`
- `prefetch(AUTOTUNE)`

### Why tf.data?

- Efficient batch loading
- Optimized GPU utilization
- Real-time augmentation
- Scalable for large datasets

---

# 8. 🧠 CNN Model Architecture

Architecture includes:

- Rescaling layer
- 6× Conv2D layers
- MaxPooling after each
- Flatten
- Dense(64)
- Dense(3) with softmax

Designed to learn:

✔ textures

✔ blight patterns

✔ shape distortions

---

# 9. 📊 Training, Evaluation & Results

Training:

- 50 epochs
- Adam optimizer
- sparse_categorical_crossentropy

Results:

- **Training Accuracy:** ~99%
- **Validation Accuracy:** 97–98%
- **Test Accuracy:** ~98%

Model generalizes extremely well.

---

# 10. 💾 Model Saving & Versioning

Automatically detects latest version and saves model into:

```
saved_models/
    └── 1/
        ├── model.keras
        ├── weights.weights.h5

```

Versioning ensures:

✔ Traceability

✔ Reproducibility

✔ MLOps readiness

---

# 11. ⚡ FastAPI Backend (main-aug.py)

### Your backend loads:

### **Direct Keras Model + Weights**

Key features:

- `/ping` → Health check
- `/predict` → Image upload → Preprocessing → CNN inference
- Returns JSON:

```json
{
  "class": "Late Blight",
  "confidence": 0.982
}

```

Image preprocessing:

- Converts file → RGB
- Resizes → (256,256)
- Normalizes → 0–1
- Adds batch dimension

---

# 12. 💻 Frontend (React.js)

Built using:

- Material UI
- Dropzone image uploader
- Axios (API calls)
- Live preview of uploaded image
- Displays prediction + confidence

User flow:

1. Drag & drop leaf image
2. React sends it to FastAPI
3. API returns disease & confidence
4. UI shows results cleanly

---

# 13. 📱 Mobile App (Future Integration)

Model conversion:

✔ TensorFlow Lite (TFLite)

✔ Optimized for real-time mobile inference

Will power Android/iOS app for farmers.

---

# 14. 🚀 Deployment Guide

### Local Deployment:

✔ Python virtual environment

✔ FastAPI server

✔ React app

### Docker (Optional):

- TensorFlow Serving container
- Exposes port 8501
- Hot model reload

### Cloud Deployment:

✔ Vercel for frontend

✔ GCP or AWS for backend

✔ Cloud Storage for models

---

# 15. 📁 Project Folder Structure

```
Potato-disease-classification/
│
├── api/
│   ├── main-aug.py
│   ├── main.py
│   └── main-tf-serving.py
│
├── saved_models/
│   └── 1/
│       ├── model.keras
│       └── weights.weights.h5
│
├── frontend/
│   ├── src/
│   └── public/
│
├── mobile-app/   (future)
├── static/
├── requirements.txt
├── README.md
└── ...

```

---

# 16. ▶ How to Run the Project

## 1️⃣ Setup environment

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

```

## 2️⃣ Run FastAPI server

```
cd api
python main-aug.py

```

Server runs at:

👉 [http://localhost:8000](http://localhost:8000/)

Docs UI:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

## 3️⃣ Run React App

```
cd frontend
npm install
npm start

```

Open:

👉 [http://localhost:3000](http://localhost:3000/)

---

# 17. 🌟 Future Improvements

- Add more crops (tomato, cotton, maize)
- Add bounding-box leaf detection
- On-device inference with TFLite
- Multilingual farmer UI
- Better augmentation (cutmix, color jitter)
- Integrate with farmer advisory system

---

# 18. 👨‍💻 Credits

**Developed by:** Gaurav Singh

**Domain:** Deep Learning, MLOps, Agritech

**Architecture:** TensorFlow + FastAPI + React.js
