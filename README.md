# Hybrid Quantum-Classical CNN for Medical Image Diagnosis

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/jayanthk82/ats)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository presents a **Hybrid Quantum-Classical Convolutional Neural Network (Q-CNN)** for automated thoracic disease diagnosis from medical images. By combining the feature extraction strength of classical deep learning with the high-dimensional expressivity of Quantum Machine Learning (QML), the project demonstrates a scalable pipeline for real-world datasets such as **CheXpert**.

---

## 🎯 Core Objectives

- Integrate **PennyLane** quantum computing with high-performance **JAX / PyTorch** workflows.
- Build an efficient data pipeline for large-scale medical imaging datasets.
- Design a hybrid architecture that improves diagnostic performance while reducing computational overhead.

---

## 🧠 Technical Architecture

The system follows a **Parallel Residual Hybrid Design** to stabilize training and mitigate the *Barren Plateau* issue common in quantum circuits.

### 1) Classical Feature Extraction

- Uses a pre-trained **ResNet18** to extract **512-dimensional** feature vectors from **224×224** X-ray images.
- Pre-computes image features to decouple heavy processing.
- Accelerates hybrid training by ~**50×**.

### 2) Quantum Path — *“Booster”*

- Compresses classical features into **10 qubits** using dense layers.
- Applies **AngleEmbedding** and **StronglyEntanglingLayers**.
- Explores correlations in a **2¹⁰ Hilbert space** to enhance classification.

### 3) Classical Path — *“Safety Net”*

- Runs a parallel dense network to match baseline neural performance.
- Merges outputs using a residual connection:

\[
\text{Final Output} = \text{Classical Output} + \text{Quantum Output}
\]

---

## ⚙️ Key Features & Techniques

- **High-Performance Computing**: JAX JIT compilation with XLA optimization for faster quantum simulation.
- **Uncertainty Handling**: Recall optimization by mapping uncertain labels to potentially diseased states.
- **Class Imbalance Handling**: **Focal Loss** (γ = 2.0) to prioritize hard minority classes.
- **Clinical Tuning**: Dynamic decision thresholding to maximize **F1-Score** per disease risk.

---

## 📊 Performance Results

| Metric                         | Result   |
|-------------------------------|----------|
| Best Validation Accuracy      | **86.85%** |
| Macro AUROC                   | **0.7035** |
| Recall (Atelectasis)         | **91%** |
| Recall (Edema)               | **95%** |

> **Note:** Designed as a high-sensitivity first-pass screening model, prioritizing **Recall** to minimize missed pathologies.

---

## 🚀 Future Scope

- Scale to more qubits and deeper quantum layers.
- Deploy on real **Quantum Processing Units (QPUs)**.
- Extend to **3D CT scan** data processing.

---
## Datset 
- Dataset available at https://www.kaggle.com/datasets/ashery/chexpert
