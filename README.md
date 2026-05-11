# AI-Powered-IDPS

## Overview

AI-Powered-IDPS is a Hybrid Intrusion Detection and Prevention System that combines traditional signature-based detection using Suricata with Machine Learning and Deep Learning models for intelligent real-time network traffic analysis.

The project integrates:

* Suricata IDS
* XGBoost
* Random Forest
* LSTM
* Flask API
* Real-Time Traffic Monitoring

to create an AI-powered cybersecurity framework capable of detecting malicious network activity.

---

# Features

* Real-time network traffic monitoring
* AI-based attack detection
* Hybrid IDS architecture
* Suricata integration
* XGBoost intrusion detection
* Random Forest classifier
* LSTM deep learning model
* Flask backend API
* Live traffic prediction
* Graph generation and analysis

---

# Project Architecture

Internet Traffic
↓
Suricata IDS
↓
eve.json Logs
↓
AI Detection Engine
↓
XGBoost / Random Forest / LSTM
↓
Flask Backend API
↓
Frontend Dashboard

---

# Technologies Used

## Cybersecurity

* Suricata IDS

## Machine Learning

* Scikit-learn
* XGBoost
* TensorFlow/Keras

## Backend

* Flask

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib

---

# Dataset

Dataset Used:
CICIDS2017 Dataset

Source:
https://www.unb.ca/cic/datasets/ids-2017.html

---

# Models Implemented

## Random Forest

Used for fast and stable intrusion detection with strong interpretability.

## LSTM

Used for deep learning-based sequential traffic pattern analysis.

## XGBoost

Used for high-performance real-time attack detection.

---

# Real-Time Detection

The system captures real-time network traffic using Suricata and analyzes the generated events using AI models to classify traffic as:

* NORMAL
* ATTACK

---

# API Endpoints

## Home Route

GET /

## Prediction Route

POST /predict

Example JSON Input:

```json
{
  "features": [0,0,0,0]
}
```

Example JSON Output:

```json
{
  "prediction": "NORMAL"
}
```

---

# Graphs

The project includes:

* Confusion Matrix
* Accuracy Graphs
* Feature Importance Graphs
* Label Distribution Graphs

---

# Future Improvements

* Multi-class attack classification
* Explainable AI (XAI)
* Cloud deployment
* Blockchain-based alert logging
* Live dashboard integration
* Federated learning
* Transformer-based intrusion detection

---

# Contributors

* Soham Kadam — AI & Machine Learning
* Sanskriti — Frontend & Backend
* Arun — Cybersecurity & Blockchain
* Kunal — Cloud Infrastructure

---

# License

MIT License
