import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cleaner = SimpleImputer(strategy="mean")

raw_data = {
    'Amount': [
        12.5, 250.0, 5000.0, 120.0, None, 45.0, 15000.0, 75.0, 300.0, 20.0,
        6000.0, 5.0, 80.0, None, 200.0, 400.0, 150.0, 7000.0, 30.0, 100.0,
        25000.0, 15.0, 90.0, 3500.0, None, 60.0, 110.0, 9000.0, 27.0, 500.0
    ],
    'Is_International': [
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0
    ],
    'Is_Late_Night': [
        0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 1, 0, 0, 1, 1, 0
    ],
    'Is_Fraud': [
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0
    ]
}
frame = pd.DataFrame(raw_data)
frame[["Amount"]] = cleaner.fit_transform(frame[["Amount"]])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
x = frame[["Amount", "Is_International", "Is_Late_Night"]]
y = frame[["Is_Fraud"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train.values.ravel())
score = model.score(x_test, y_test)

print(f"Model Prediction Score: {score * 100:.2f}%")
print(f"Feature Importance Score: {model.n_features_in_}")

# Manual entering of data
amount = float(input("Amount: "))
international = True if input("Is International? (Y/N): ").lower() == "y" else False
late_night = True if input("Late Night? (Y/N): ").lower() == "y" else False
data_frame = pd.DataFrame([
    {
        "Amount": amount,
        "Is_International": international,
        "Is_Late_Night": late_night,
    }
], columns=["Amount", "Is_International", "Is_Late_Night"])
print(data_frame)

# Predict
prediction = model.predict_proba(data_frame)
print(f"Safe transaction: {prediction[0][0] * 100:.2f}%")
print(f"Fraud transaction: {prediction[0][1] * 100:.2f}%")
