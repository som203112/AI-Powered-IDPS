import pandas as pd
import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD ALL CSV FILES
# =========================

path = "data/MachineLearningCSV/*.csv"

csv_files = glob.glob(path)

dataframes = []

print("Loading CSV files...")

for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"Loaded: {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Merge all datasets
df = pd.concat(dataframes, ignore_index=True)

print("\nDataset Shape:", df.shape)

# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip()

# =========================
# REMOVE INVALID VALUES
# =========================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

print("Dataset after cleaning:", df.shape)

# =========================
# LABEL ENCODING
# =========================

# BENIGN = 0
# ATTACK = 1

df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nLabel Distribution:")
print(df["Label"].value_counts())

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("Label", axis=1)

# Remove non-numeric columns if any
X = X.select_dtypes(include=[np.number])

y = df["Label"]

print("\nFeature Shape:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "model/attack_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved successfully!")

# =========================
# CONFUSION MATRIX GRAPH
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("graphs/RandomForest_confusion_matrix.png")

# =========================
# LABEL DISTRIBUTION GRAPH
# =========================

plt.figure(figsize=(6, 5))

sns.countplot(x=df["Label"])

plt.title("Attack vs Benign Distribution")

plt.savefig("graphs/RandomForest_label_distribution.png")

print("\nGraphs saved in graphs/ folder")