# ======================= HIGGS BINARY CLASSIFICATION (ONE-FILE VERSION) =======================
# This script:
# 1) Loads the ATLAS Higgs Challenge dataset from a .csv.gz file
# 2) Cleans & standardizes features
# 3) Splits into Train/Val/Test (80/10/10)
# 4) Trains a dense neural network for binary classification (signal=1 vs background=0)
# 5) Evaluates with accuracy, confusion matrix, classification report, and precision-recall curve
# 6) Saves the final model to disk
# ----------------------------------------------------------------------------------------------

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from keras.src.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, f1_score, average_precision_score)
from sklearn.metrics import roc_curve, auc

# ----------------------------- Reproducibility -----------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------- Data Loader -----------------------------
def load_higgs_data_gz(file_path="atlas-higgs-challenge-2014-v2.csv.gz"):
    """
    Robust loader for the ATLAS Higgs dataset stored as .csv.gz.
    """
    p = Path(file_path).expanduser()
    candidates = [
        p,
        Path.cwd() / p.name,
        Path(__file__).parent / p.name,
        Path.home() / "Desktop" / p.name,
    ]
    found = next((c for c in candidates if c.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"Could not find '{p.name}'. Place the file in your project folder/Desktop."
        )

    df = pd.read_csv(found, compression="gzip")
    df = df.replace({"t": 1, "f": 0})

    if "Label" not in df.columns:
        raise KeyError("Column 'Label' not found in dataset.")
    y = df["Label"].map({"s": 1, "b": 0}).astype(int).values

    drop_cols = [c for c in ["EventId", "Weight", "Label"] if c in df.columns]
    X_df = df.drop(columns=drop_cols)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    return X_scaled, y

# ----------------------------- Model Builder -----------------------------
def build_binary_mlp(input_dim: int):
    """
    Dense neural network for binary classification.
    Uses sigmoid + binary_crossentropy.
    """
    from tensorflow.keras.layers import Input

    model = Sequential([
        Input(shape=(input_dim,), name="input_layer"),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense1"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense2"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(1, activation="sigmoid", name="output_layer")
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ----------------------------- Training Helper -----------------------------
def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Fit with EarlyStopping and ModelCheckpoint.
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history

# ----------------------------- Evaluation Helper -----------------------------
def evaluate_binary(model, X_test, y_test):
    """
    Evaluate model on test set: accuracy, confusion matrix, classification report, PR curve.
    """
    # Test Loss & Accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

    # Predictions
    y_proba = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)

    # Confusion Matrix & Classification Report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["background", "signal"])
    print("\nClassification Report:\n", cr)

    # Plot Confusion Matrix
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["background", "signal"],
                yticklabels=["background", "signal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP={ap:.3f}, F1={f1:.3f}")

    # Belirli threshold noktalarını işaretle
    for t in [0.9, 0.5, 0.1]:
        idx = np.argmin(np.abs(thresholds - t))  # en yakın threshold’u bul
        plt.scatter(recall[idx], precision[idx], marker="o", label=f"thr={t:.1f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with thresholds")
    plt.legend()
    plt.show()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")  # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    # 1) Load data
    X, y = load_higgs_data_gz("atlas-higgs-challenge-2014-v2.csv.gz")

    # 2) Split into Train / Val / Test (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 3) Build model
    model = build_binary_mlp(input_dim=X_train.shape[1])
    model.summary()

    # 4) Train
    _ = train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # 5) Evaluate
    evaluate_binary(model, X_test, y_test)

    # 6) Save final model
    model.save("final_dense_higgs_model.keras")
    print("Saved -> final_dense_higgs_model.keras")

# ======================= END OF SCRIPT =======================

