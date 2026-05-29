# Face Verification & Identification
> **CIE 555 — Neural Networks and Deep Learning**  
> University of Science and Technology, Zewail City · Spring 2026

---

## Overview

A full deep learning pipeline for **face verification** on the LFW (Labeled Faces in the Wild) dataset — determining whether two face images belong to the same person.

The project implements and compares three architectures across two loss paradigms, following the progression from a custom CNN baseline to DeepFace-style and FaceNet-style transfer learning models.

---

## Central Question

> **Given two face images, do they belong to the same person?**

---

## Dataset

| Property | Details |
|----------|---------|
| Source | [LFW — Labeled Faces in the Wild](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) |
| Identities (after filtering) | ~1,000+ individuals |
| Filter criteria | ≥ 12 images per identity |
| Identity cap | ≤ 30 images per identity (class balance fix) |
| Split | Identity-level 80/20 train/val — validation contains fully unseen people |

---

## Pipeline

### Preprocessing
- **MTCNN face detection**: automatically detects and crops the face region from each image with a 15% margin, eliminating background, hair, and clothing noise
- Images resized to **160×160** (FaceNet standard)
- Normalized to [0, 1]
- Identity-level train/val split ensures no identity leakage into validation

### Models

| Model | Backbone | Approach | Loss |
|-------|----------|----------|------|
| **Custom CNN** | 4-block CNN (32→64→128→256) | Pairs | Binary Cross-Entropy |
| **Custom CNN** | 4-block CNN (32→64→128→256) | Triplets | Triplet Loss |
| **DeepFace-style** | ResNet50V2 (pretrained) | Pairs | Binary Cross-Entropy |
| **FaceNet-style** | InceptionResNetV2 (pretrained) | Triplets | Triplet Loss + Hard Mining |

### Key Implementation Details
- **Both loss functions implemented manually from scratch** (no library loss functions used)
- **L2-normalized embeddings** on unit hypersphere for all models — consistent distance scale across batches
- **Two-phase training** for pretrained backbones: frozen warm-up → partial unfreeze fine-tuning
- **Semi-hard triplet mining** (FaceNet-style): selects hard positives (max intra-class distance) and semi-hard negatives (violates margin but avoids collapse)
- **F1-calibrated thresholds**: optimal decision threshold found by sweeping F1 score, not fixed at 0.5

### Evaluation
- Validation accuracy (threshold-calibrated)
- AUC-ROC (threshold-independent)
- Best F1 score + optimal threshold per model
- Distance distribution plots (same vs different person overlap)
- Full inference pipeline: takes two image paths → returns distance, decision, visual display

---

## Tech Stack

- **Python 3**
- **TensorFlow / Keras** — model building and training
- **MTCNN** — face detection and cropping
- **NumPy** — embedding operations, hard mining
- **scikit-learn** — AUC-ROC, F1, train/val split
- **Matplotlib / Seaborn** — all visualizations

---

## File Structure

```
├── Face_Verification_Identification.ipynb   # Full pipeline
├── requirements.txt                         # Dependencies
├── report.pdf
└── README.md
```

---

> **Note:** MTCNN requires `lz4` for loading pretrained weights. Make sure it is installed before running.

---
