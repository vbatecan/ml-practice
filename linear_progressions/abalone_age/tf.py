import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.metrics import Recall, Precision
from keras.layers import Dropout

frame = pd.read_csv("data.csv")
scaler = StandardScaler()

sex_dummies = pd.get_dummies(frame[["sex"]], dtype="int")
x = frame.drop("rings", axis=1)
y = frame["rings"]

x = x.drop("sex", axis=1)
x = pd.concat([x, sex_dummies], axis=1)
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_ts = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test_ts = tf.convert_to_tensor(x_test, dtype=tf.float32)

y_train_ts = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test_ts = tf.convert_to_tensor(y_test, dtype=tf.float32)

model = Sequential([
    Dense(32, activation="relu", input_shape=(x_train_ts.shape[1],)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(loss="mse", optimizer="adam", metrics=["mae"])
model.fit(x_train_ts, y_train_ts, epochs=100, batch_size=32, validation_split=0.2)

score = model.evaluate(x_test_ts, y_test_ts)
print(f"Score: {score}")

# Let's Predict
predictions = model.predict(x_test_ts)
print(predictions)
