import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==============================
# LOAD DATASET
# ==============================

print("Loading CSV files...")

csv_files = glob.glob("data/MachineLearningCSV/*.csv")

dataframes = []

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

# ==============================
# CLEAN DATA
# ==============================

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Replace inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
df.dropna(inplace=True)

print("Dataset after cleaning:", df.shape)

# ==============================
# LABEL ENCODING
# ==============================

# Convert labels into binary
df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nLabel Distribution:")
print(df["Label"].value_counts())

# ==============================
# FEATURES & LABELS
# ==============================

X = df.drop("Label", axis=1)

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

y = df["Label"]

print("\nFeature Shape:", X.shape)

# ==============================
# FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "model/lstm_scaler.pkl")

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# RESHAPE FOR LSTM
# ==============================

# LSTM expects 3D input:
# (samples, timesteps, features)

X_train = X_train.reshape(
    (X_train.shape[0], 1, X_train.shape[1])
)

X_test = X_test.reshape(
    (X_test.shape[0], 1, X_test.shape[1])
)

# ==============================
# BUILD LSTM MODEL
# ==============================

print("\nBuilding LSTM model...")

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(1, X_train.shape[2]),
        return_sequences=False
    )
)

model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining LSTM model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)

print("\nTraining completed!")

# ==============================
# PREDICTIONS
# ==============================

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix - LSTM")

plt.colorbar()

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha='center',
            va='center',
            color='black'
        )

plt.savefig("graphs/lstm_confusion_matrix.png")

# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Train Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("LSTM Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("graphs/lstm_accuracy.png")

# ==============================
# SAVE MODEL
# ==============================

model.save("model/lstm_model.h5")

print("\nLSTM model saved successfully!")

print("\nGraphs saved in graphs/ folder")