import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read CSV
frame = pd.read_csv("data.csv")
frame["volume"] = frame["length"] * frame["diameter"] * frame["height"]
frame["weight_ratio"] = frame["whole_weight"] / (frame["length"] * frame["diameter"])

sex_dummies = pd.get_dummies(frame[["sex"]])

x = frame.drop(["rings", "sex"], axis=1)
x = pd.concat([x, sex_dummies], axis=1)
y = frame[["rings"]]

model = LinearRegression()

# Split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

# Score
score = model.score(x_test, y_test)
print(f"Prediction score: {score:.2f}%")
