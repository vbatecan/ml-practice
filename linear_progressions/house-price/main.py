import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model = LinearRegression()
scaler = StandardScaler()
data = {
    "size": [
        1500,
        3000,
        1200,
        2500
    ],
    "bedrooms": [
        3,
        4,
        2,
        3
    ],
    "location": [
        "Suburb",
        "City center",
        "Suburb",
        "Countryside",
    ],
    "age": [
        10,
        5,
        15,
        8
    ],
    "price": [
        100000,
        200000,
        90000,
        150000
    ]
}

frame = pd.DataFrame(data)
x = frame[["size", "bedrooms", "location", "age"]]
x_dummies = pd.get_dummies(x[["location"]], dtype="int")
# Replace 'location' and concat dummies
x = pd.concat([x, x_dummies], axis=1)
x = x.drop("location", axis=1)

y = frame[["price"]]
print(x.columns, x.values)
print(y)

model.fit(x, y)
data_input = pd.DataFrame([[3000, 6, 5, 1, 0, 0]], columns=x.columns)

prediction = model.predict(data_input)
print(f"The house price is: {prediction[0][0]:.2f}")
