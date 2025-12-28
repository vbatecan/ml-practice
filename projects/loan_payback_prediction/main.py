import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

frame = pd.read_csv("train.csv")

scaler = MinMaxScaler()

# Dummies
gender_d = pd.get_dummies(frame["gender"])
marital_status_d = pd.get_dummies(frame["marital_status"])
education_level_d = pd.get_dummies(frame["education_level"])
employment_status_d = pd.get_dummies(frame["employment_status"])
loan_purpose_d = pd.get_dummies(frame["loan_purpose"])
grade_subgrade_d = pd.get_dummies(frame["grade_subgrade"])

# Preprocessing
x = frame.drop(
    columns=["loan_paid_back", "gender", "marital_status", "education_level", "employment_status", "loan_purpose",
             "grade_subgrade"], axis=1)
y = frame["loan_paid_back"]

# Drop and add dummies
x = pd.concat([x, gender_d, marital_status_d, education_level_d, employment_status_d, loan_purpose_d, grade_subgrade_d],
              axis=1)
x = scaler.fit_transform(x)

# model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Convert to tensor
x_train_tf = tf.convert_to_tensor(x, dtype=tf.float32)
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)

y_train_tf = tf.convert_to_tensor(y, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

model = Sequential([
    Dense(64, activation="relu", input_shape=(x_train_tf.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.compile(optimizer="adam", loss="binary_cross entropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping])
model.save("model.keras")

evaluate = model.evaluate(x_test, y_test)
print(f"Evaluation: {evaluate}")
print(f"Test loss: {evaluate[0]}")
print(f"Test accuracy: {evaluate[1]}")
predictions = model.predict(x_test)
