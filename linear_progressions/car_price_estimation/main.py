import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "year": [
        2018,
        2020,
        2017,
        2019
    ],
    "brand": [
        "Toyota",
        "Honda",
        "Ford",
        "BMW"
    ],
    "engine_size": [
        1.5,
        2.0,
        1.8,
        3.0
    ],
    "fuel_type": [
        "Gasoline",
        "Diesel",
        "Gasoline",
        "Gasoline"
    ],
    "price": [
        15000,
        20000,
        13000,
        30000
    ]
}

model = LinearRegression()
frame = pd.DataFrame(data)
brand_dummies = pd.get_dummies(frame[["brand"]], dtype="int")
fuel_type_dummies = pd.get_dummies(frame[["fuel_type"]], dtype="int")
print(brand_dummies)
print(fuel_type_dummies)

x = pd.concat([frame[["year", "engine_size"]], brand_dummies, fuel_type_dummies], axis=1)
print(x.head())

y = frame[["price"]]

model.fit(x, y)
print(model.coef_)
print(model.intercept_)

print(x.columns)
# Data input
data_in = pd.DataFrame([
    [2012, 1.5, 0, 0, 0, 1, 0, 1]
], columns=x.columns)

prediction = model.predict(data_in)
print(f"The estimated price is: {prediction[0][0]:.2f}")