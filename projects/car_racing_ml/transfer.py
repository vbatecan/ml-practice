from keras.src.saving.saving_api import load_model
from keras.src.layers.regularization.dropout import Dropout
from keras.src.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection._split import train_test_split
from keras.src.layers.core.dense import Dense
from keras.src.models.sequential import Sequential
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

DATA = "combined_data.csv"
MODEL = "model.h5"
SAVE_MODEL = "transferred.h5"
USE_SMOTE = False

frame = pd.read_csv(DATA)
scaler = StandardScaler()

# Preprocessing
x = frame.drop(columns=["W", "A", "S", "D", "Handbrake", "Collision"], axis=1)
y = frame[["W", "A", "S", "D", "Handbrake"]]

x = scaler.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

if USE_SMOTE:
    y_train_combined = y_train.astype(str).apply(lambda x: "".join(x), axis=1)

    print(y_train_combined.value_counts())

    smote = SMOTE()
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
    patience=3,
    restore_best_weights=True,
)

model = load_model(MODEL)

for layer in model.layers[:-1]:
    layer.trainable = False

for layer in model.layers[-5:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),  # 10x smaller than default
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=150, callbacks=[early_stop], validation_split=0.2, validation_data=[x_test, y_test])
model.save(SAVE_MODEL)

score = model.evaluate(x_test, y_test)
print("Score: ", score)

predictions = model.predict(x_test)
print(predictions)
