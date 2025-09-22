# ======================= HIGGS BINARY CLASSIFICATION (higgs.csv version with graphs) =======================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from keras.src.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, f1_score, average_precision_score,
                             roc_curve, auc)

# ----------------------------- Reproducibility -----------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------- Data Loader -----------------------------
def load_higgs_csv(file_path="higgs.csv"):
    p = Path(file_path).expanduser()
    if not p.exists():
        desktop_path = Path.home() / "Desktop" / p.name
        if desktop_path.exists():
            p = desktop_path
        else:
            raise FileNotFoundError(f"Could not find {file_path} or {desktop_path}")

    df = pd.read_csv(p, low_memory=False)

    if "is_boson" not in df.columns:
        raise KeyError("Column 'is_boson' not found in dataset.")

    # '?' → NaN
    df = df.replace("?", np.nan)

    # Feature set → numeric + NaN→0
    X_df = df.drop(columns=["is_boson"]).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Label
    y = df["is_boson"].astype(int).values

    # Normalize 0-1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    print("NaN count:", np.isnan(X_scaled).sum(), "Inf count:", np.isinf(X_scaled).sum())

    return X_scaled, y

# ----------------------------- Model Builder -----------------------------
def build_binary_mlp(input_dim: int):
    model = Sequential([
        Input(shape=(input_dim,), name="input_layer"),
        Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
                  metrics=["accuracy"])
    return model

# ----------------------------- Training Helper -----------------------------
def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), weights))
    print("Class weights:", class_weights)

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
        class_weight=class_weights,
        verbose=1
    )
    return history

# ----------------------------- Evaluation Helper -----------------------------
def evaluate_binary(model, X_test, y_test):
    """
    Evaluate model on test set: accuracy, confusion matrix, classification report,
    PR curve (best threshold işaretli), and ROC curve.
    """
    # --- Accuracy & Loss ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

    # --- Predictions ---
    y_proba = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)

    # --- Confusion Matrix & Report ---
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["background", "signal"])
    print("\nClassification Report:\n", cr)

    # --- Confusion Matrix Plot ---
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["background", "signal"],
                yticklabels=["background", "signal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.show()

    # --- Precision-Recall Curve ---
    from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    # En iyi threshold (F1 maksimize eden)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print("Best threshold (F1):", best_threshold,
          "Precision:", precision[best_idx],
          "Recall:", recall[best_idx],
          "F1:", f1_scores[best_idx])

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP={ap:.3f}, F1@0.5={f1:.3f}")
    plt.scatter(recall[best_idx], precision[best_idx], color="red",
                label=f"Best threshold={best_threshold:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with optimal threshold")
    plt.legend()
    plt.show()

    # --- ROC Curve ---
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    X, y = load_higgs_csv("higgs.csv")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = build_binary_mlp(input_dim=X_train.shape[1])
    model.summary()

    _ = train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    evaluate_binary(model, X_test, y_test)

    model.save("final_dense_higgs_model.keras")
    print("Saved -> final_dense_higgs_model.keras")

# ======================= END OF SCRIPT =======================
