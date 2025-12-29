from keras.src.layers.normalization.batch_normalization import BatchNormalization
from keras.src.layers.core.input_layer import Input
from imblearn.over_sampling import SMOTE
from keras.src.layers.regularization.dropout import Dropout
from keras.src.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection._split import train_test_split
from keras.src.layers.core.dense import Dense
from keras.src.models.sequential import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

frame = pd.read_csv("data.csv")
scaler = StandardScaler()
USE_SMOTE = False

# Preprocessing
x = frame.drop(columns=["W", "A", "S", "D", "Handbrake", "Collision"], axis=1)
y = frame[["W", "A", "S", "D", "Handbrake"]]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.01, random_state=42
)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

if USE_SMOTE:
    y_train_combined = y_train.astype(str).apply(lambda x: "".join(x), axis=1)

    print(y_train_combined)

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
    patience=5,
    restore_best_weights=True,
)

model = Sequential(
    [
        Input(shape=(x_train.shape[1],)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(5, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, callbacks=[early_stop], validation_split=0.2)
model.save("model.h5")

score = model.evaluate(x_test, y_test)
print("Score: ", score)

predictions = model.predict(x_test)
print(predictions)
