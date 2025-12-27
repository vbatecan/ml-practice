import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = {
    "hours": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
    ],
    "scores": [
        48,
        55,
        63,
        72,
        79,
        85,
        91
    ]
}

scaler = StandardScaler()
model = LinearRegression()
frame = pd.DataFrame(data)
x = frame[["hours"]]
y = frame[["scores"]]

x_scaled = scaler.fit_transform(x)
model.fit(x, y)
print(x_scaled)

# hours
hour = int(input("Enter hour: "))
prediction = model.predict(pd.DataFrame([[hour]], columns=["hours"]))
print(f"Prediction: {prediction[0][0]:.2f}")
