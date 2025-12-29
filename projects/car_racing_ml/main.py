from keras.src.layers.normalization.batch_normalization import BatchNormalization
from keras.src.layers.core.input_layer import Input
from imblearn.over_sampling import SMOTE
from keras.src.layers.regularization.dropout import Dropout
from keras.src.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection._split import train_test_split
from keras.src.layers.core.dense import Dense
from keras.src.models.sequential import Sequential
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    classification_report,
    multilabel_confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_SENSORS = 10

frame = pd.read_csv("combined_data.csv")

scaler = StandardScaler()
USE_SMOTE = False

# Log-transform the original R{i} columns
for i in range(NUMBER_OF_SENSORS):
    frame[f"R{i}"] = np.log1p(frame[f"R{i}"])

# Remove rows where W == 1 and Collision == 1
frame = frame[~((frame["W"] == 1) & (frame["Collision"] == 1))]
print(f"Data shape after removing W=1 & Collision=1: {frame.shape}")

# Downsample "W"
w_only = frame[
    (frame["W"] == 1) & (frame[["A", "S", "D", "Handbrake"]].sum(axis=1) == 0)
]
actions = frame[frame[["A", "S", "D", "Handbrake"]].sum(axis=1) > 0]
w_only_sample = w_only.sample(frac=0.15, random_state=42)

# Combine
balanced_frame = pd.concat([w_only_sample, actions]).sample(frac=1).reset_index(drop=True)

# Preprocessing
# x = frame.drop(columns=["W", "A", "S", "D", "Handbrake", "Collision"], axis=1)
# y = frame[["W", "A", "S", "D", "Handbrake"]]
print(w_only_sample.shape[0])
print(balanced_frame.shape[0])
print(frame.shape[0])

x = balanced_frame.drop(columns=["W", "A", "S", "D", "Handbrake", "Collision"], axis=1)
y = balanced_frame[["W", "A", "S", "D", "Handbrake"]]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42
)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# sys.exit(0)

if USE_SMOTE:
    y_train_combined = y_train.astype(str).apply(lambda x: "".join(x), axis=1)

    print(y_train_combined)

    smote = SMOTE(k_neighbors=3)
    x_resampled, y_resampled_combined = smote.fit_resample(x_train, y_train_combined)

    y_resampled = pd.DataFrame(
        [list(map(int, list(i))) for i in y_resampled_combined],
        columns=["W", "A", "S", "D", "Handbrake"],
    )

    print(y_resampled)
    x_train = x_resampled
    y_train = y_resampled


# Convert to tensor
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

model = Sequential(
    [
        Input(shape=(x_train.shape[1],)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(64),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dense(5, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    x_train,
    y_train,
    epochs=100,
    callbacks=[early_stop],
    validation_split=0.15,
    batch_size=32,
)
model.save("model_3.h5")

score = model.evaluate(x_test, y_test)
print("Score: ", score)

# Show metrics like F1 Score and Confusion Matrix
LABELS = ["W", "A", "S", "D", "Handbrake"]

# Predict + threshold
x_test_pred_probs = model.predict(x_test)
x_test_pred = (x_test_pred_probs >= 0.5).astype(int)

# Core metrics
f1_macro = f1_score(y_test, x_test_pred, average="macro", zero_division=0)
f1_micro = f1_score(y_test, x_test_pred, average="micro", zero_division=0)
f1_weighted = f1_score(y_test, x_test_pred, average="weighted", zero_division=0)
class_report = classification_report(y_test, x_test_pred, target_names=LABELS, zero_division=0)
confusion_matrix = multilabel_confusion_matrix(y_test, x_test_pred)

print("F1 (macro):", f1_macro)
print("F1 (micro):", f1_micro)
print("F1 (weighted):", f1_weighted)
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrices:\n", confusion_matrix)

# Optional: ROC-AUC per label
for i, label in enumerate(LABELS):
    try:
        # Check if both classes (0 and 1) are present in y_test for this label
        if len(np.unique(y_test[:, i])) > 1:
            auc = roc_auc_score(y_test[:, i], x_test_pred_probs[:, i])
            print(f"AUC ({label}): {auc:.3f}")
        else:
            print(f"AUC ({label}): Not defined (only one class present)")
    except Exception as e:
        print(f"AUC ({label}): Error {e}")

# ---- VISUALIZE CONFUSION MATRICES ----
fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 4))
if len(LABELS) == 1:
    axes = [axes]

for ax, label, cm in zip(axes, LABELS, confusion_matrix):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(label)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.suptitle("Confusion Matrices per Label")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("confusion_matrices.png")
print("Saved confusion_matrices.png")

# ---- LABEL-WISE F1 BAR CHART ----
f1_per_label = f1_score(y_test, x_test_pred, average=None, zero_division=0)
plt.figure(figsize=(8, 5))
plt.bar(LABELS, f1_per_label)
plt.ylabel("F1 Score")
plt.ylim(0, 1.1)
plt.title("Label-wise F1 Scores")
for i, v in enumerate(f1_per_label):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.savefig("f1_scores.png")
print("Saved f1_scores.png")
plt.show()
