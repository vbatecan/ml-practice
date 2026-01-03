import matplotlib as plt
import pandas as pd
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from sklearn.preprocessing._data import MinMaxScaler, StandardScaler

MODEL = "model.keras"
TEST_DATA = "test.csv"

frame = pd.read_csv(TEST_DATA)
scaler = MinMaxScaler()

gender_d = pd.get_dummies(frame["gender"])
marital_status_d = pd.get_dummies(frame["marital_status"])
education_level_d = pd.get_dummies(frame["education_level"])
employment_status_d = pd.get_dummies(frame["employment_status"])
loan_purpose_d = pd.get_dummies(frame["loan_purpose"])
grade_subgrade_d = pd.get_dummies(frame["grade_subgrade"])


# Preprocessing
features = frame.drop(
    columns=["gender", "marital_status", "education_level", "employment_status", "loan_purpose",
             "grade_subgrade"], axis=1)

# Drop and add dummies
features = pd.concat([features, gender_d, marital_status_d, education_level_d, employment_status_d, loan_purpose_d, grade_subgrade_d],
              axis=1)
features = scaler.fit_transform(features)

model = load_model(MODEL)

predictions = model.predict(features)
predictions = predictions.ravel()

to_save = pd.DataFrame({
    "id": frame["id"],
    "loan_paid_back": predictions
})

# Save to csv
to_save.to_csv("predictions2.csv", index=False)