# ============================================
# xgboost_train.py
# AI-Based Intrusion Detection System
# ============================================

import os
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================
# CREATE REQUIRED FOLDERS
# ============================================

os.makedirs("model", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# ============================================
# LOAD DATASET
# ============================================

print("Loading CSV files...\n")

csv_files = glob.glob(
    "data/MachineLearningCSV/*.csv"
)

dataframes = []

for file in csv_files:

    try:

        df = pd.read_csv(file)

        print(f"Loaded: {file}")

        dataframes.append(df)

    except Exception as e:

        print(f"Error loading {file}")
        print(e)

# ============================================
# MERGE DATAFRAMES
# ============================================

df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\nDataset Shape:", df.shape)

# ============================================
# CLEAN COLUMN NAMES
# ============================================

df.columns = df.columns.str.strip()

# ============================================
# REMOVE NULL + INF VALUES
# ============================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(inplace=True)

print(
    "Dataset after cleaning:",
    df.shape
)

# ============================================
# CONVERT LABELS
# ============================================

df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nLabel Distribution:\n")

print(df["Label"].value_counts())

# ============================================
# SPLIT FEATURES AND LABELS
# ============================================

X = df.drop("Label", axis=1)

y = df["Label"]

# ============================================
# SAVE FEATURE NAMES
# ============================================

feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    "model/feature_names.pkl"
)

print("\nFeature names saved successfully!")

# ============================================
# FEATURE SHAPE
# ============================================

print("\nFeature Shape:", X.shape)

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed!")

print("Training Samples:", X_train.shape[0])

print("Testing Samples:", X_test.shape[0])

# ============================================
# FEATURE SCALING
# ============================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ============================================
# SAVE SCALER
# ============================================

joblib.dump(
    scaler,
    "model/xgb_scaler.pkl"
)

print("Scaler saved successfully!")

# ============================================
# TRAIN XGBOOST MODEL
# ============================================

print("\nTraining XGBoost model...\n")

model = XGBClassifier(

    n_estimators=100,

    max_depth=6,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed!")

# ============================================
# PREDICTIONS
# ============================================

y_pred = model.predict(
    X_test_scaled
)

# ============================================
# ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.title("XGBoost Confusion Matrix")

plt.savefig(
    "graphs/xgboost_confusion_matrix.png"
)

plt.close()

print("\nConfusion Matrix saved!")

# ============================================
# FEATURE IMPORTANCE GRAPH
# ============================================

importance = model.feature_importances_

indices = np.argsort(importance)[-10:]

plt.figure(figsize=(10, 6))

plt.barh(
    range(len(indices)),
    importance[indices]
)

plt.yticks(
    range(len(indices)),
    [feature_names[i] for i in indices]
)

plt.xlabel("Importance")

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(
    "graphs/xgboost_feature_importance.png"
)

plt.close()

print("Feature Importance graph saved!")

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(
    model,
    "model/xgboost_model.pkl"
)

print("\nXGBoost model saved successfully!")

# ============================================
# FINAL MESSAGE
# ============================================

print("\n===================================")

print("✅ XGBoost Training Completed!")

print("===================================")